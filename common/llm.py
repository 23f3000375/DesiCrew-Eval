"""Provider-agnostic LLM client with automatic fallback (no vendor SDK needed - plain HTTPS).

Every provider below speaks the OpenAI-compatible chat-completions protocol, including Google's Gemini endpoint,
so one code path handles text, images (vision), JSON mode and tool calling for all of them.

Chain order (only providers whose key is set are used):  gemini -> groq -> openrouter -> mistral -> openai -> ollama
Within a provider, several models are tried (auto-discovered from the provider's /models list, or set <PROVIDER>_MODELS).

Behaviour when something fails (this is what the 503 "model is overloaded" problem needed):
  * 503 / 429 / 5xx / dropped connection -> that model goes on a short cooldown and the NEXT model/provider is tried
    immediately (no long sleeping). The primary is retried first again once its cooldown ends.
  * unsupported JSON mode / tools / images -> that capability is switched off for that model and the call is retried.
  * bad key -> that provider is disabled for the rest of the run, with a clear message.
  * only if every usable model is cooling down does the client wait (capped), up to LLM_TOTAL_TIMEOUT seconds.
"""
import base64
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- public helpers


@dataclass
class Img:
    data: bytes
    mime: str = "image/jpeg"


def image_part(data: bytes, mime: str = "image/jpeg") -> Img:
    return Img(data, mime)


def parse_json(text: str):
    """Parse JSON from a model response, tolerating ```json fences, <think> blocks and stray prose."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


# --------------------------------------------------------------------------- provider table
PROVIDERS = {
    "gemini": dict(base="https://generativelanguage.googleapis.com/v1beta/openai", keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"), interval=4.0),
    "groq": dict(base="https://api.groq.com/openai/v1", keys=("GROQ_API_KEY",), interval=2.5),
    "openrouter": dict(base="https://openrouter.ai/api/v1", keys=("OPENROUTER_API_KEY",), interval=3.0),
    "mistral": dict(base="https://api.mistral.ai/v1", keys=("MISTRAL_API_KEY",), interval=2.0),
    "openai": dict(base="https://api.openai.com/v1", keys=("OPENAI_API_KEY",), interval=1.0),
    "ollama": dict(base="http://localhost:11434/v1", keys=(), interval=0.0),
}
DEFAULT_ORDER = ["gemini", "groq", "openrouter", "mistral", "openai", "ollama"]
NON_CHAT = r"image|tts|live|audio|embed|robotics|computer|native|imagen|veo|gemma-?2|learnlm|whisper|guard|playai|moderation|rerank|dall|transcri"
VISION_NAME = r"llama-4|scout|maverick|vision|-vl|pixtral|gemma-3|gemini|gpt-4o|gpt-4\.1|gpt-5|mistral-(small|medium|large)"


@dataclass
class Candidate:
    provider: str
    model: str
    vision: bool = True
    tools: bool = True
    json_mode: bool = True
    disabled: bool = False
    cooldown_until: float = 0.0
    hard_fails: int = 0

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class ProviderError(Exception):
    def __init__(self, kind, message, cooldown=0.0):
        super().__init__(message)
        self.kind, self.cooldown = kind, cooldown


def _env(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _truthy(v):
    return str(v).lower() in {"1", "true", "yes", "y"}


def _version(model_id: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", model_id)
    return float(m.group(1)) if m else 0.0


# --------------------------------------------------------------------------- the client
class LLM:
    def __init__(self, model: str | None = None, providers: list | None = None, min_interval: float | None = None):
        """model: optional 'provider:model' (or bare model name for the first provider) to put first in the chain."""
        order = providers or [p.strip() for p in _env("LLM_PROVIDERS", ",".join(DEFAULT_ORDER)).split(",") if p.strip()]
        self.keys, self.bases = {}, {}
        for p in order:
            if p not in PROVIDERS:
                continue
            spec = PROVIDERS[p]
            key = next((os.getenv(k) for k in spec["keys"] if os.getenv(k)), None)
            if p == "ollama":
                key = "ollama" if _env("OLLAMA_MODEL") else None
            if key:
                self.keys[p] = key
                self.bases[p] = _env(f"{p.upper()}_BASE_URL", spec["base"]).rstrip("/")
        if not self.keys:
            raise RuntimeError(
                "No LLM API key found. Put at least one in .env: GEMINI_API_KEY (https://aistudio.google.com/apikey), "
                "GROQ_API_KEY (https://console.groq.com/keys), OPENROUTER_API_KEY (https://openrouter.ai/keys), "
                "MISTRAL_API_KEY, OPENAI_API_KEY, or OLLAMA_MODEL for a local model."
            )
        self.order = [p for p in order if p in self.keys]
        self.forced = model
        self.interval_override = min_interval if min_interval is not None else (
            float(_env("LLM_MIN_INTERVAL")) if _env("LLM_MIN_INTERVAL") else None)
        self.timeout = float(_env("LLM_REQUEST_TIMEOUT", 90))
        self.total_timeout = float(_env("LLM_TOTAL_TIMEOUT", 300))
        self._cands: list[Candidate] | None = None
        self._last_call: dict = {}
        self.stats: Counter = Counter()
        self.last_used: Candidate | None = None
        self.verbose = _truthy(_env("LLM_VERBOSE", "1"))

    # ------------------------------------------------------------ candidates / discovery
    @property
    def model(self) -> str:
        if self.last_used:
            return self.last_used.label
        return "auto (" + " > ".join(self.order) + ")"

    def candidates(self) -> list[Candidate]:
        if self._cands is None:
            self._cands = self._build_candidates()
        return self._cands

    def _log(self, msg):
        if self.verbose:
            print(f"  [llm] {msg}", flush=True)

    def _build_candidates(self):
        out = []
        for p in self.order:
            try:
                out += self._candidates_for(p)
            except Exception as e:  # noqa: BLE001
                self._log(f"could not build model list for {p} ({str(e)[:80]}); using defaults")
                out += [Candidate(p, m) for m in self._default_models(p)]
        if self.forced:
            prov, name = self.forced.split(":", 1) if ":" in self.forced else (self.order[0], self.forced)
            if prov in self.keys:
                out.insert(0, Candidate(prov, name, vision=True))
        if not out:
            raise RuntimeError("No models available for the configured providers.")
        return out

    @staticmethod
    def _default_models(p):
        return {"gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"],
                "groq": ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"],
                "openrouter": ["google/gemma-3-27b-it:free"],
                "mistral": ["mistral-small-latest"], "openai": ["gpt-4o-mini"],
                "ollama": [_env("OLLAMA_MODEL", "llama3.2-vision")]}[p]

    def _list_models(self, p):
        r = requests.get(f"{self.bases[p]}/models", headers=self._headers(p), timeout=25)
        r.raise_for_status()
        return r.json().get("data", [])

    def _candidates_for(self, p):
        override = _env(f"{p.upper()}_MODELS")
        vis_flag = _env(f"{p.upper()}_VISION")
        if p == "ollama":
            return [Candidate(p, m, vision=_truthy(_env("OLLAMA_VISION", "1")), tools=True) for m in self._default_models(p)]
        if override:
            names = [m.strip() for m in override.split(",") if m.strip()]
            return [Candidate(p, m, vision=_truthy(vis_flag) if vis_flag else bool(re.search(VISION_NAME, m))) for m in names]
        if p in ("mistral", "openai"):
            return [Candidate(p, m, vision=True) for m in self._default_models(p)]
        data = self._list_models(p)
        if p == "gemini":
            ids = [str(d.get("id", "")).replace("models/", "") for d in data]
            ids = [i for i in ids if "gemini" in i and "flash" in i and not re.search(NON_CHAT, i) and not re.search(r"-exp|thinking", i)]
            ids.sort(key=lambda i: ("preview" in i, "lite" in i, "latest" in i, -_version(i), i))
            return [Candidate(p, i, vision=True) for i in ids[:5]] or [Candidate(p, m) for m in self._default_models(p)]
        if p == "groq":
            ids = [str(d.get("id", "")) for d in data]
            ids = [i for i in ids if not re.search(NON_CHAT, i)]
            vis = [i for i in ids if re.search(VISION_NAME, i)]
            txt = [i for i in ids if i not in vis and re.search(r"llama-3\.3-70b|gpt-oss|qwen|kimi|llama-3\.1-70b", i)]
            return [Candidate(p, i, vision=True) for i in vis[:3]] + [Candidate(p, i, vision=False) for i in txt[:3]]
        if p == "openrouter":
            cands = []
            for d in data:
                mid = str(d.get("id", ""))
                pr = d.get("pricing") or {}
                free = mid.endswith(":free") or (str(pr.get("prompt")) == "0" and str(pr.get("completion")) == "0")
                if not free or re.search(NON_CHAT, mid):
                    continue
                vision = "image" in ((d.get("architecture") or {}).get("input_modalities") or [])
                tools = "tools" in (d.get("supported_parameters") or [])
                pref = bool(re.search(r"gemma-3|llama-4|qwen|mistral|gemini|llama-3\.3", mid))
                cands.append((not vision, not pref, mid, Candidate(p, mid, vision=vision, tools=tools)))
            cands.sort(key=lambda t: t[:3])
            vis = [c for *_, c in cands if c.vision][:3]
            txt = [c for *_, c in cands if not c.vision][:3]
            return vis + txt
        return [Candidate(p, m) for m in self._default_models(p)]

    # ------------------------------------------------------------ wire format
    def _headers(self, p):
        h = {"Authorization": f"Bearer {self.keys[p]}", "Content-Type": "application/json"}
        if p == "openrouter":
            h.update({"HTTP-Referer": "https://localhost", "X-Title": "desicrew-ds-eval"})
        return h

    @staticmethod
    def to_messages(contents, system=None):
        """contents: str | list[str | Img]  ->  OpenAI-format messages."""
        parts = [contents] if isinstance(contents, (str, Img)) else list(contents)
        blocks = []
        for x in parts:
            if isinstance(x, Img):
                b64 = base64.b64encode(x.data).decode()
                blocks.append({"type": "image_url", "image_url": {"url": f"data:{x.mime};base64,{b64}"}})
            else:
                blocks.append({"type": "text", "text": str(x)})
        content = blocks[0]["text"] if len(blocks) == 1 and blocks[0]["type"] == "text" else blocks
        msgs = [{"role": "system", "content": system}] if system else []
        return msgs + [{"role": "user", "content": content}]

    @staticmethod
    def _has_images(messages):
        return any(isinstance(m.get("content"), list) and any(b.get("type") == "image_url" for b in m["content"]) for m in messages)

    @staticmethod
    def _sanitize(messages, provider):
        out = []
        for m in messages:
            m = dict(m)
            if provider != "gemini":
                m.pop("extra_content", None)
                if m.get("tool_calls"):
                    m["tool_calls"] = [{k: v for k, v in tc.items() if k != "extra_content"} for tc in m["tool_calls"]]
            if m.get("role") == "assistant" and m.get("content") is None:
                m["content"] = ""
            out.append(m)
        return out

    def _throttle(self, p):
        gap = self.interval_override if self.interval_override is not None else PROVIDERS[p]["interval"]
        wait = gap - (time.time() - self._last_call.get(p, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last_call[p] = time.time()

    @staticmethod
    def _cooldown_from(r, default):
        ra = r.headers.get("retry-after")
        if ra and ra.replace(".", "", 1).isdigit():
            return min(max(float(ra), 5), 120)
        m = re.search(r"retry in ([\d.]+)s|try again in ([\d.]+)s|retryDelay\D{0,5}(\d+)s", r.text)
        if m:
            return min(max(float(next(g for g in m.groups() if g)) + 1, 5), 120)
        return default

    def _classify(self, r):
        s, text = r.status_code, (r.text or "")[:600]
        low = text.lower()
        if s in (401, 403) or (s == 400 and "api key" in low and ("invalid" in low or "not valid" in low)):
            return ProviderError("auth", f"{s} {text[:160]}")
        if s == 404:
            return ProviderError("model_missing", f"404 {text[:160]}")
        if s == 429:
            return ProviderError("transient", f"429 rate limit / quota: {text[:120]}", self._cooldown_from(r, 60))
        if s in (500, 502, 503, 504, 529):
            return ProviderError("transient", f"{s} {text[:120]}", self._cooldown_from(r, 30 if s == 503 else 20))
        if s in (400, 413, 422):
            if "tool_use_failed" in low or "failed_generation" in low:
                return ProviderError("tool_failed", text[:160], 3)
            if "response_format" in low or "json_object" in low or "json mode" in low:
                return ProviderError("unsupported_json", text[:160])
            if "tool" in low and re.search(r"not support|unsupported|does not support", low):
                return ProviderError("unsupported_tools", text[:160])
            if re.search(r"image|vision|multimodal|modalit", low) and re.search(r"not support|unsupported|does not support|invalid", low):
                return ProviderError("unsupported_vision", text[:160])
            return ProviderError("other", f"{s} {text[:200]}", 45)
        return ProviderError("other", f"{s} {text[:200]}", 45)

    def _call(self, c: Candidate, messages, tools, temperature, json_mode):
        p = c.provider
        self._throttle(p)
        body = {"model": c.model, "messages": self._sanitize(messages, p)}
        if tools:
            body["tools"] = tools
        if temperature is not None and not (p == "gemini" and "gemini-3" in c.model):
            body["temperature"] = temperature
        if json_mode and c.json_mode:
            body["response_format"] = {"type": "json_object"}
        if p == "openrouter":
            body["max_tokens"] = 4096
        try:
            r = requests.post(f"{self.bases[p]}/chat/completions", headers=self._headers(p), json=body, timeout=self.timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise ProviderError("transient", f"{type(e).__name__}: {str(e)[:120]}", 15)
        if r.status_code != 200:
            raise self._classify(r)
        try:
            msg = r.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError, TypeError):
            raise ProviderError("bad_output", f"unexpected response shape: {r.text[:120]}", 5)
        if not (msg.get("content") or msg.get("tool_calls")):
            raise ProviderError("bad_output", "empty response", 5)
        return msg

    # ------------------------------------------------------------ main entry points
    def chat(self, messages, *, tools=None, temperature=0.0, json_mode=False, tag=None) -> dict:
        """Run one chat completion with fallback. Returns the assistant message dict (content / tool_calls)."""
        need_vision, need_tools = self._has_images(messages), bool(tools)
        deadline = time.time() + self.total_timeout
        errors: list[str] = []
        while True:
            cands = [c for c in self.candidates() if not c.disabled and (c.vision or not need_vision) and (c.tools or not need_tools)]
            if not cands:
                raise RuntimeError("No usable model" + (" with vision" if need_vision else "") + (" with tool calling" if need_tools else "")
                                   + " left. Problems seen:\n  " + "\n  ".join(dict.fromkeys(errors[-8:])) +
                                   "\nCheck your keys in .env (run: python common/doctor.py).")
            if all(c.hard_fails >= 2 for c in cands):
                raise RuntimeError("Every model rejected the request:\n  " + "\n  ".join(dict.fromkeys(errors[-8:])))
            now = time.time()
            ready = [c for c in cands if c.cooldown_until <= now]
            if not ready:
                if now > deadline:
                    raise RuntimeError("All models are busy/rate-limited and the time limit was reached:\n  " + "\n  ".join(dict.fromkeys(errors[-8:])))
                wait = min(c.cooldown_until for c in cands) - now
                self._log(f"all models cooling down; waiting {min(max(wait, 1), 15):.0f}s")
                time.sleep(min(max(wait, 1), 15))
                continue
            c = ready[0]
            try:
                msg = self._call(c, messages, tools, temperature, json_mode)
                c.hard_fails = 0
                self.stats[c.label] += 1
                if self.last_used is not c:
                    self._log(f"using {c.label}")
                self.last_used = c
                return msg
            except ProviderError as e:
                errors.append(f"{c.label}: {e.kind}: {e}")
                if e.kind == "auth":
                    for x in self.candidates():
                        if x.provider == c.provider:
                            x.disabled = True
                    self._log(f"{c.provider}: API key rejected ({str(e)[:80]}) - provider disabled for this run")
                elif e.kind == "model_missing":
                    c.disabled = True
                    self._log(f"{c.label}: model not found - skipping")
                elif e.kind == "unsupported_json":
                    c.json_mode = False
                    self._log(f"{c.label}: no JSON mode - continuing without it")
                elif e.kind == "unsupported_tools":
                    c.tools = False
                elif e.kind == "unsupported_vision":
                    c.vision = False
                    self._log(f"{c.label}: no image support - skipping for image requests")
                else:
                    c.cooldown_until = time.time() + e.cooldown
                    if e.kind == "other":
                        c.hard_fails += 1
                    self._log(f"{c.label}: {e.kind} ({str(e)[:90]}) -> trying next model")
                if time.time() > deadline and all(x.cooldown_until > time.time() or x.disabled for x in self.candidates()):
                    raise RuntimeError("Time limit reached. Problems seen:\n  " + "\n  ".join(dict.fromkeys(errors[-8:])))

    def text(self, contents, *, system=None, temperature=0.0, tag=None, **_) -> str:
        return self.chat(self.to_messages(contents, system), temperature=temperature, tag=tag)["content"] or ""

    def json(self, contents, *, system=None, temperature=0.0, tag=None, **_):
        messages = self.to_messages(contents, system)
        last = None
        for attempt in range(4):
            msg = self.chat(messages, temperature=temperature, json_mode=True, tag=tag)
            try:
                return parse_json(msg.get("content") or "")
            except (json.JSONDecodeError, ValueError) as e:
                last = e
                if self.last_used:
                    self.last_used.cooldown_until = time.time() + 5  # nudge the next attempt onto another model
                self._log(f"reply was not valid JSON ({str(e)[:60]}); retrying")
        raise RuntimeError(f"model did not return valid JSON after 4 attempts: {last}")

    # ------------------------------------------------------------ diagnostics
    def probe(self, c: Candidate, messages, **kw):
        """Call ONE specific candidate, no fallback (used by doctor.py)."""
        return self._call(c, messages, kw.get("tools"), kw.get("temperature", 0.0), kw.get("json_mode", False))


def retry_call(fn, max_retries=6, label="llm"):
    """Generic retry helper for transient network errors (kept for compatibility)."""
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{label} failed after {max_retries} retries: {last}")
