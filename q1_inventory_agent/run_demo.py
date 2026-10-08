"""Run a scripted conversation and save a markdown transcript (evidence for the submission).
    python q1_inventory_agent/run_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q1_inventory_agent.agent import InventoryAgent  # noqa: E402

QUESTIONS = [
    "Give me a quick overview of the inventory: how many products, total units on hand and total stock value?",
    "Which 5 products hold the most inventory value, and what share of the total is that?",
    "Which products sold through the highest percentage of the stock they had available?",
    "Do the hand-in-stock numbers reconcile with opening + purchased - sold? List any that don't and by how much.",
    "What is inventory turnover? Look up the standard formula and calculate it for the whole catalog from this sheet.",
    "Chart units sold for the top 10 products.",
]

if __name__ == "__main__":
    agent = InventoryAgent()
    out = ["# Inventory agent - demo transcript", "", ""]
    for q in QUESTIONS:
        print(f"\nYOU> {q}")
        r = agent.ask(q)
        out += [f"## Q: {q}", ""]
        for s in r.steps:
            if s.kind == "code":
                out += ["<details><summary>code run</summary>", "", "```python", s.input.strip(), "```", "",
                        "```", (s.error or s.output).strip(), "```", "</details>", ""]
            else:
                out += [f"_web search:_ `{s.input}`", ""]
        out += [r.text, ""]
        print(f"AGENT> {r.text}")
    out[1] = f"_models used (calls per model): {dict(agent.llm.stats)}_"
    p = Path(__file__).with_name("demo_transcript.md")
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved {p}")
