"""Check which providers/models work with YOUR keys:   python common/doctor.py   (add --all to test every model)

Tests per model: plain JSON reply, reading text from an image (vision), and tool calling.
Run this before the pipeline: it tells you in ~1 minute what the fallback chain will actually be able to do.
"""
import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common.llm import LLM, ProviderError, image_part, parse_json  # noqa: E402


def test_image():
    im = Image.new("RGB", (520, 140), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=56)
    except TypeError:
        font = ImageFont.load_default()
    d.text((20, 35), "ID 4821-X", fill="black", font=font)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


ADD_TOOL = [{"type": "function", "function": {"name": "add", "description": "Add two integers",
             "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]


def probe(llm, c, kind):
    t0 = time.time()
    try:
        if kind == "json":
            m = llm.probe(c, llm.to_messages('Reply with exactly this JSON: {"ok": true}'), json_mode=True)
            ok = parse_json(m.get("content") or "").get("ok") is True
        elif kind == "vision":
            msgs = llm.to_messages([image_part(test_image(), "image/png"), 'Read the text in the image. Return JSON {"text": "<what it says>"}'])
            m = llm.probe(c, msgs, json_mode=True)
            ok = "4821" in str(parse_json(m.get("content") or "").get("text", ""))
        else:
            m = llm.probe(c, llm.to_messages("What is 17 + 25? You must use the add tool."), tools=ADD_TOOL)
            ok = bool(m.get("tool_calls"))
        return ("PASS" if ok else "FAIL (wrong answer)"), time.time() - t0
    except ProviderError as e:
        return f"FAIL ({e.kind}: {str(e)[:70]})", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return f"FAIL ({type(e).__name__}: {str(e)[:70]})", time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="test every discovered model, not just the first per provider")
    a = ap.parse_args()
    llm = LLM()
    print("providers with a key:", ", ".join(llm.order))
    cands = llm.candidates()
    print("fallback chain (in order):")
    for c in cands:
        print(f"   {c.label:60s} vision={c.vision!s:5} tools={c.tools}")
    seen, works = set(), {"json": [], "vision": [], "tools": []}
    print("\nlive tests:")
    for c in cands:
        if not a.all and c.provider in seen:
            continue
        seen.add(c.provider)
        for kind in ("json", "vision", "tools"):
            if kind == "vision" and not c.vision:
                print(f"   {c.label:55s} {kind:7s} skipped (text-only model)")
                continue
            res, dt = probe(llm, c, kind)
            print(f"   {c.label:55s} {kind:7s} {res}  ({dt:.1f}s)")
            if res == "PASS":
                works[kind].append(c.label)
    print("\nsummary")
    for k, v in works.items():
        print(f"   {k:7s}: {', '.join(v) if v else 'NONE - add another provider key (see .env.example)'}")
    if not works["vision"]:
        print("\n!! No vision-capable model works. Q2 ingestion and Q3 need one. Add GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY.")
