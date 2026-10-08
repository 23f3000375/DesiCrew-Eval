"""Web search tool that does not depend on any LLM vendor: DuckDuckGo (ddgs package), then Wikipedia as a backup."""
import re

import requests


def _ddg(query, n):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # older package name
    hits = DDGS().text(query, max_results=n) or []
    return [(h.get("title", ""), h.get("body", ""), h.get("href", "")) for h in hits]


def _wikipedia(query, n):
    r = requests.get("https://en.wikipedia.org/w/api.php", timeout=15, headers={"User-Agent": "desicrew-eval/1.0"}, params=dict(
        action="query", generator="search", gsrsearch=query, gsrlimit=n, prop="extracts|info", inprop="url",
        exintro=1, explaintext=1, exlimit=n, format="json"))
    r.raise_for_status()
    pages = sorted((r.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    return [(p.get("title", ""), re.sub(r"\s+", " ", p.get("extract", ""))[:600], p.get("fullurl", "")) for p in pages]


def web_search(query: str, max_results: int = 4):
    """-> (text, sources, error). Never raises."""
    errs = []
    for name, fn in (("duckduckgo", _ddg), ("wikipedia", _wikipedia)):
        try:
            hits = [h for h in fn(query, max_results) if h[1] or h[0]]
            if hits:
                text = "\n".join(f"- {t}: {b}" for t, b, _ in hits)[:2500]
                return text, [f"{t} - {u}" for t, _, u in hits if u], None
            errs.append(f"{name}: no results")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__} {str(e)[:80]}")
    return ("Search is unavailable right now (" + "; ".join(errs) + "). Answer from general knowledge and say clearly that "
            "you could not verify it online."), [], "; ".join(errs)
