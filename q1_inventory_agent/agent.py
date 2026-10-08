"""Inventory data agent: writes + executes pandas code, searches the web for definitions/context, and explains
results in plain English. Multi-turn. Uses the fallback LLM chain from common/llm.py, so an overloaded model or provider
does not stop the conversation (the whole message history is re-sent, so it can switch model mid-conversation).

Tools exposed to the model (OpenAI-style function calling):
    run_python(code)   -> executes in a sandboxed subprocess against DataFrame `df`
    web_search(query)  -> DuckDuckGo, with Wikipedia as backup
"""
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.llm import LLM  # noqa: E402
from q1_inventory_agent import sandbox  # noqa: E402
from q1_inventory_agent import search as search_mod  # noqa: E402
from q1_inventory_agent.data_loader import load_inventory, schema_text  # noqa: E402

SYSTEM = """You are an inventory analyst assistant chatting with a business user about ONE spreadsheet that is already
loaded as a pandas DataFrame named `df`. The user is not technical: explain results in plain English.

{schema}

How to work:
1. NEVER state a number, ranking or comparison from memory or by eyeballing the preview. Compute it with the
   `run_python` tool and base the answer on the printed output. Print what you need (use print() or end with an expression).
   `df`, `pd`, `np` and `plt` are already available; only pandas/numpy/math/statistics/datetime/matplotlib-style imports work.
   Each run_python call starts fresh from the original `df` (changes you make are not kept), so redo any prep you need.
2. If the code errors, read the error, fix it and run again. Do not give up after one failure.
3. For questions about definitions, formulas, benchmarks or business context (e.g. "what is a good sell-through rate",
   "what does reorder point mean"), use the `web_search` tool, then apply the definition to the data with code. Do not
   search for things the data itself answers.
4. If a chart helps, draw it with matplotlib and save it as 'chart.png' (one chart per run, readable labels).
5. Check the data against itself when it is relevant to the question (e.g. whether recorded totals match their
   components) and mention real discrepancies you find. Never invent columns or facts the sheet does not contain.
   If the question cannot be answered from this data, say what is missing.
6. Final answer format: lead with the direct answer in 1-2 sentences, then at most a few short supporting points with the key
   numbers (USD with thousands separators, units as whole numbers). State any assumption you made (e.g. how you defined a
   metric). No code in the final answer, no markdown tables unless asked. Keep it conversational.
7. Remember earlier turns of this conversation: resolve follow-ups like "and the second one?" or "why?" from context."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python (pandas) code against the inventory DataFrame `df` and return what it printed. "
                       "`df`, `pd`, `np`, `plt` are pre-loaded. Use print() to show results. Save charts as 'chart.png'.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for a definition, formula, benchmark or business context; returns short snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "concise search query"}},
                       "required": ["query"]}}},
]


@dataclass
class Step:
    kind: str  # "code" | "search"
    input: str
    output: str = ""
    error: str | None = None
    charts: list = field(default_factory=list)
    sources: list = field(default_factory=list)


@dataclass
class Reply:
    text: str
    steps: list
    charts: list


def parse_args(raw) -> dict:
    """Tool-call arguments arrive as a JSON string; weaker models sometimes emit slightly broken JSON for code."""
    if isinstance(raw, dict):
        return raw
    for strict in (True, False):
        try:
            v = json.loads(raw, strict=strict)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r'"code"\s*:\s*"(.*)"\s*\}?\s*$', raw or "", re.S)
    if m:
        try:
            return {"code": m.group(1).encode("utf-8").decode("unicode_escape")}
        except UnicodeDecodeError:
            return {"code": m.group(1)}
    return {}


class InventoryAgent:
    def __init__(self, llm: LLM | None = None, df=None, max_tool_calls=8, searcher=None):
        self.llm = llm or LLM()
        self.df = df if df is not None else load_inventory()
        self.searcher = searcher or search_mod.web_search
        self.max_tool_calls = max_tool_calls
        self.system = {"role": "system", "content": SYSTEM.format(schema=schema_text(self.df))}
        self.history: list[dict] = []
        self.steps: list[Step] = []

    def reset(self):
        self.history, self.steps = [], []

    # ---------------- tools
    def run_python(self, code: str) -> str:
        res = sandbox.run_python(code, self.df)
        self.steps.append(Step("code", code, res["stdout"], res["error"], res["charts"]))
        if res["error"]:
            return f"ERROR:\n{res['error']}\n(stdout so far:\n{res['stdout']})"
        out = res["stdout"] or "(no output - did you forget print()?)"
        if res["charts"]:
            out += "\n[chart saved and shown to the user]"
        return out

    def web_search(self, query: str) -> str:
        text, sources, err = self.searcher(query)
        self.steps.append(Step("search", query, text, err if sources == [] and err else None, sources=sources))
        return text + ("\nSources: " + "; ".join(sources[:3]) if sources else "")

    def _dispatch(self, name, args):
        if name == "run_python":
            code = args.get("code")
            return self.run_python(code) if code else "ERROR: missing 'code' argument. Call run_python again with valid JSON."
        if name == "web_search":
            q = args.get("query")
            return self.web_search(q) if q else "ERROR: missing 'query' argument."
        return f"ERROR: unknown tool '{name}'. Available tools: run_python, web_search."

    # ---------------- chat
    def ask(self, question: str) -> Reply:
        self.steps = []
        msgs = [self.system] + self.history + [{"role": "user", "content": question}]
        final = None
        for i in range(self.max_tool_calls + 1):
            msg = self.llm.chat(msgs, tools=TOOLS if i < self.max_tool_calls else None, temperature=0.2)
            calls = msg.get("tool_calls") or []
            if not calls:
                final = (msg.get("content") or "").strip()
                break
            keep = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "extra_content")}
            keep.setdefault("role", "assistant")
            msgs.append(keep)
            for tc in calls:
                fn = tc.get("function", {})
                out = self._dispatch(fn.get("name"), parse_args(fn.get("arguments")))
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", f"call_{i}"), "content": out[:4000]})
        if not final:
            final = "I couldn't finish that analysis - please try rephrasing the question."
        self.history = (msgs[1:] + [{"role": "assistant", "content": final}])[-60:]
        while self.history and self.history[0]["role"] != "user":  # never start history mid tool-exchange
            self.history.pop(0)
        charts = [c for s in self.steps for c in s.charts]
        return Reply(final, list(self.steps), charts)


if __name__ == "__main__":
    a = InventoryAgent()
    print("type 'exit' to quit")
    while True:
        q = input("\nyou> ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        r = a.ask(q)
        for s in r.steps:
            tag = "CODE" if s.kind == "code" else "SEARCH"
            print(f"\n  [{tag}] {s.input.strip()[:600]}\n  -> {(s.error or s.output).strip()[:400]}")
        print(f"\n[{a.llm.model}] agent> {r.text}")
