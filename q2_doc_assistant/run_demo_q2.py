"""Run the scripted 13-turn demo conversation and save the transcript.
    python q2_doc_assistant/run_demo.py [--mode vision|tesseract] [--extra-forms]
Make sure you have looked at kb/sections.json first (python q2_doc_assistant/kb.py) - citations are only as good as that file.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.llm import LLM  # noqa: E402
from kb import KnowledgeBase, ingest  # noqa: E402
from session import SupportAssistant  # noqa: E402

# Designed to exercise: follow-ups, pronoun resolution, a deliberate repeat, topic switches, return to an earlier topic,
# an out-of-scope question, and a meta/summary turn.
TURNS = [
    "What is the Assignment Request Form used for?",
    "Who is the assignee in this form, and how are they related to the policyholder?",
    "What reason was given for the assignment, and what proof is attached?",
    "Is the assignment conditional or absolute?",
    "Sorry, can you remind me who the assignee is?",                                    # deliberate repeat
    "Who witnessed the endorsement?",
    "Different question - what plan and premium are in the proposal form?",             # topic switch
    "What does the proposal declaration say if I misstate facts?",
    "And what if my health changes after I submit the proposal?",                       # follow-up
    "Do I have to agree to be contacted on WhatsApp?",
    "What is HDFC Life's claim settlement ratio?",                                      # not in the documents
    "Back to the assignment form - who signs if the assignee is a minor?",              # return to earlier topic
    "Can you summarise everything we've covered so far?",                               # meta
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vision", "tesseract"], default="vision")
    ap.add_argument("--extra-forms", action="store_true")
    a = ap.parse_args()
    llm = LLM()
    kb = KnowledgeBase(ingest(llm, a.mode, a.extra_forms))
    bot = SupportAssistant(kb, llm)
    for q in TURNS:
        r = bot.ask(q)
        print(f"\n[{r['turn']}] YOU> {q}\n    ({r['relation']} · {r['topic']})\n    BOT> {r['answer']}")
        for s in r["sources"]:
            print(f"        [{s['n']}] {s['label']}")
    out = HERE / "demo_transcript.md"
    out.write_text(bot.transcript_md() + f"\n\n_models used (calls per model): {dict(llm.stats)}_\n", encoding="utf-8")
    print(f"\nSaved {out}")
