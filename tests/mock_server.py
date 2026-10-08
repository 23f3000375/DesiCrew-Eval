"""A local fake of an OpenAI-compatible LLM API, for testing the real client over real HTTP.

Each MockProvider is its own server and can be switched into failure modes at runtime:
   ok | 503 | 429 | auth | no_json | no_vision | alternate (every 2nd request is a 503)
The 'brain' answers deterministically for each task in this repo (Q1 tool calls, Q2 rewrite/answer/transcribe,
Q3 classify/extract/locate/zoom) using the Q3 answer key, so end-to-end results can be checked exactly.
"""
import base64
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "q3_doc_pipeline"))
sys.path.insert(0, str(ROOT))

from schemas import DOC_TYPES  # noqa: E402

Q3 = ROOT / "q3_doc_pipeline"
KEY = {k: v for k, v in json.loads((Q3 / "answer_key.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
STEM_TO_TYPE = {
    "Aadhar": "aadhaar", "ID": "pan", "ChatGPT Image May 2, 2026, 03_43_11 PM": "driving_licence",
    "ChatGPT Image May 2, 2026, 03_52_54 PM": "passport", "ECS": "nach_ecs", "Fatca": "fatca",
    "Illustration": "benefit_illustration", "Moral": "moral_hazard", "split": "multiple_policies",
    "suitability": "suitability_profiler",
}
TYPE_TO_FILE = {STEM_TO_TYPE[Path(f).stem]: f for f in KEY}
LABEL_TO_TYPE = {v["label"]: k for k, v in DOC_TYPES.items()}
FIXTURE_ROWS = json.loads((ROOT / "q2_doc_assistant" / "tests" / "fixture_sections.json").read_text(encoding="utf-8"))
REWRITES = {
    "What is the assignment request form for?": ("purpose of the assignment request form", "assignment form purpose", "new_topic"),
    "Who signs if the assignee is a minor?": ("who signs when the assignee is a minor appointee signature", "minor assignee", "follow_up"),
    "What is the claim settlement ratio?": ("claim settlement ratio", "claim ratio", "new_topic"),
}


def _thumb(im):
    return list(im.convert("L").resize((16, 16)).tobytes())


_THUMBS = {}


def nearest_file(img_bytes):
    if not _THUMBS:
        for f in KEY:
            _THUMBS[f] = _thumb(Image.open(Q3 / "data" / f))
    t = _thumb(Image.open(io.BytesIO(img_bytes)))
    return min(_THUMBS, key=lambda f: sum(abs(a - b) for a, b in zip(_THUMBS[f], t)))


def _parts(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return c, []
    text = " ".join(b["text"] for b in c if b["type"] == "text")
    imgs = [base64.b64decode(b["image_url"]["url"].split(",", 1)[1]) for b in c if b["type"] == "image_url"]
    return text, imgs


def brain(body, state):
    msgs, tools = body["messages"], body.get("tools")
    last = msgs[-1]
    if tools:
        return _tool_brain(msgs, tools)
    text, imgs = _parts(last)
    temp = body.get("temperature", 0.0) or 0.0
    if "Reply with exactly this JSON" in text:
        return {"role": "assistant", "content": '{"ok": true}'}
    if "Read the text in the image" in text:
        return {"role": "assistant", "content": '{"text": "ID 4821-X"}'}
    if "Classify this document image" in text:
        f = nearest_file(imgs[0])
        return {"role": "assistant", "content": json.dumps({"doc_type": STEM_TO_TYPE[Path(f).stem], "confidence": 0.97, "reason": "mock"})}
    m = re.search(r"Document type: (.*?)\.\n", text)
    dt = LABEL_TO_TYPE.get(m.group(1)) if m else None
    if "bounding box" in text:
        names = re.findall(r'- "(\w+)":', text)
        return {"role": "assistant", "content": json.dumps({"boxes": {n: [300, 100, 340, 500] for n in names}})}
    if "zoomed-in crops" in text:
        names = re.findall(r'- "(\w+)":', text)
        gt = KEY[TYPE_TO_FILE[dt]]
        return {"role": "assistant", "content": json.dumps({"fields": {n: {"value": gt.get(n), "confidence": 0.95} for n in names}})}
    if "Extract these fields" in text:
        gt = KEY[TYPE_TO_FILE[dt]]
        out = {}
        for f in DOC_TYPES[dt]["fields"]:
            v = gt.get(f)
            if dt == "nach_ecs" and f == "ifsc_code" and temp > 0.2:
                v = "SBINO227112"  # handwriting misread (letter O for zero) on sampled reads
            out[f] = {"value": v, "confidence": 0.9}
        return {"role": "assistant", "content": json.dumps({"fields": out})}
    if "Transcribe it into logical sections" in text:
        i = state.setdefault("transcribe", 0)
        state["transcribe"] = i + 1
        rows = FIXTURE_ROWS[(i * 4) % len(FIXTURE_ROWS):][:4]
        return {"role": "assistant", "content": json.dumps({"sections": [{"heading": r["heading"], "text": r["text"]} for r in rows]})}
    if "New user message:" in text:
        q = re.search(r'New user message: "(.*)"', text).group(1)
        s, t, r = REWRITES.get(q, (q, "general", "new_topic"))
        return {"role": "assistant", "content": json.dumps({"standalone_query": s, "topic": t, "relation": r, "previous_topic": ""})}
    if "DOCUMENT EXCERPTS" in text:
        msg = re.search(r"User's message: \"(.*)\"", text).group(1)
        if "claim settlement ratio" in msg:
            return {"role": "assistant", "content": json.dumps({"answer": "These documents do not cover that.", "citations": [], "facts_given": [], "not_covered": True})}
        t1 = re.search(r"\[(S\d+)\]", text).group(1)
        return {"role": "assistant", "content": json.dumps({"answer": f"Mock answer [{t1}].", "citations": [t1], "facts_given": ["a fact"], "not_covered": False})}
    return {"role": "assistant", "content": "{}"}


def _call(name, args, cid="call_1", extra=False):
    tc = {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if extra:
        tc["extra_content"] = {"google": {"thought_signature": "sig"}}
    return {"role": "assistant", "content": None, "tool_calls": [tc]}


def _tool_brain(msgs, tools):
    if tools[0]["function"]["name"] == "add":
        if msgs[-1]["role"] == "tool":
            return {"role": "assistant", "content": "42"}
        return _call("add", {"a": 17, "b": 25})
    idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    q = (msgs[idx]["content"] if isinstance(msgs[idx]["content"], str) else "").lower()
    called = [tc["function"]["name"] for m in msgs[idx + 1:] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]
    if not called:
        if "turnover" in q:
            return _call("web_search", {"query": "inventory turnover ratio formula"}, extra=True)
        if "chart" in q:
            return _call("run_python", {"code": "df.nlargest(10,'units_sold').plot.bar(x='product_name',y='units_sold')\nplt.tight_layout()\nplt.savefig('chart.png')\nprint('chart saved')"}, extra=True)
        return _call("run_python", {"code": "df['calc']=df.opening_stock+df.units_purchased-df.units_sold\nm=df[df.calc!=df.hand_in_stock]\nprint(len(m))\nprint(m[['product_name','hand_in_stock','calc']].head(3))"}, extra=True)
    if "turnover" in q and called == ["web_search"]:
        return _call("run_python", {"code": "print(round(df.units_sold.sum()/((df.opening_stock+df.hand_in_stock)/2).sum(),3))"}, cid="call_2", extra=True)
    return {"role": "assistant", "content": "Here is what I found: " + msgs[-1]["content"][:300]}


class MockProvider:
    def __init__(self, name, mode="ok", models=None):
        self.name, self.mode, self.requests, self.state, self.n = name, mode, [], {}, 0
        outer = self
        self.models = models or [{"id": f"{name}-model-a"}, {"id": f"{name}-model-b"}]

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    return self._send(200, {"data": outer.models})
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.n += 1
                mode = outer.mode
                has_img = any(isinstance(m.get("content"), list) and any(b.get("type") == "image_url" for b in m["content"]) for m in body["messages"])
                if mode == "alternate":
                    mode = "503" if outer.n % 2 == 1 else "ok"
                if mode == "503":
                    return self._send(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}})
                if mode == "429":
                    return self._send(429, {"error": {"message": "Rate limit reached"}}, {"retry-after": "5"})
                if mode == "auth":
                    return self._send(401, {"error": {"message": "Invalid API key"}})
                if mode == "no_json" and "response_format" in body:
                    return self._send(400, {"error": {"message": "response_format json_object is not supported by this model"}})
                if mode == "no_vision" and has_img:
                    return self._send(400, {"error": {"message": "image input is not supported for this model"}})
                if any("extra_content" in json.dumps(m) for m in body["messages"]) and outer.name != "gemini":
                    return self._send(400, {"error": {"message": "unknown field extra_content"}})
                self._send(200, {"choices": [{"index": 0, "message": brain(body, outer.state)}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
