"""Offline test of retrieval + session logic using a scripted fake LLM and a small fixture KB (no API calls)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from kb import KnowledgeBase  # noqa: E402
from session import SupportAssistant  # noqa: E402

ROWS = json.loads((Path(__file__).parent / "fixture_sections.json").read_text(encoding="utf-8"))

REWRITES = {  # message -> (standalone, topic, relation)
    "What is the assignment request form for?": ("purpose of the assignment request form", "assignment form purpose", "new_topic"),
    "What details do I need for the assignee?": ("details required about the assignee", "assignee details", "follow_up"),
    "Who signs if the assignee is a minor?": ("who signs when the assignee is a minor appointee signature", "minor assignee", "follow_up"),
    "Remind me what details the assignee needs?": ("details required about the assignee", "assignee details", "follow_up"),
    "Different question - what about misstatement of facts in the proposal?": ("misstatement suppression of material facts Section 45", "proposal misstatement", "new_topic"),
    "What is the claim settlement ratio?": ("claim settlement ratio", "claim ratio", "new_topic"),
    "Back to assignment - are there service charges?": ("policy servicing charges for assignment", "assignment charges", "return_to_earlier"),
    "Summarise what we covered": ("summary of conversation", "summary", "meta"),
}


class FakeLLM:
    def json(self, prompt, *, tag=None, **kw):
        kind, turn = tag
        if kind == "rewrite":
            q = re.search(r'New user message: "(.*)"', prompt).group(1)
            s, t, r = REWRITES[q]
            return {"standalone_query": s, "topic": t, "relation": r, "previous_topic": ""}
        # answer: simulate the model's grounded judgement for the unanswerable question, otherwise cite the first excerpt
        msg = re.search(r"User's message: \"(.*)\"", prompt).group(1)
        if "claim settlement ratio" in msg or "(no relevant excerpts found)" in prompt.split("KNOWLEDGE BASE TABLE")[0]:
            return {"answer": "These documents do not cover that.", "citations": [], "facts_given": [], "not_covered": True}
        tag1 = re.search(r"\[(S\d+)\]", prompt).group(1)
        return {"answer": f"Fake answer for turn {turn} [{tag1}].", "citations": [tag1], "facts_given": [f"fact from turn {turn}"], "not_covered": False}


def test_retrieval_recall_in_top3():
    kb = KnowledgeBase(ROWS)
    def top3(q):
        return [c.section.lower() for c, _ in kb.search(q, k=3)]
    cases = {
        "who signs if the assignee is a minor appointee": "items 4-6",
        "are there service charges for assignment": "items 4-6",        # 'service' vs 'servicing' (char n-grams)
        "misstatement of material facts Section 45": "misstatement",
        "can you contact me on WhatsApp TRAI": "communication consent",
        "is this an absolute assignment or conditional": "conditions",
        "what proof do I need to attach": "proof",
    }
    for q, want in cases.items():
        assert any(want in t for t in top3(q)), (q, top3(q))


def test_scores_cannot_gate_refusals():
    """Documented finding: an unanswerable question scores like answerable ones, so refusal must come from the LLM."""
    kb = KnowledgeBase(ROWS)
    unanswerable = kb.search("claim settlement ratio of the company", k=1, min_score=0)[0][1]
    answerable = [kb.search(q, k=1, min_score=0)[0][1] for q in
                  ["can you contact me on WhatsApp TRAI", "are there service charges for assignment"]]
    assert unanswerable > min(answerable)


def test_session_flow():
    s = SupportAssistant(KnowledgeBase(ROWS), FakeLLM())
    r1 = s.ask("What is the assignment request form for?")
    assert r1["relation"] == "new_topic" and r1["sources"] and "[1]" in r1["answer"]
    s.ask("What details do I need for the assignee?")
    r3 = s.ask("Who signs if the assignee is a minor?")
    assert "Important notes" in r3["sources"][0]["section"]
    s.ask("Remind me what details the assignee needs?")
    r5 = s.ask("Different question - what about misstatement of facts in the proposal?")
    assert r5["relation"] == "new_topic" and r5["sources"][0]["doc"] == "Proposal Ashok.pdf"
    r6 = s.ask("What is the claim settlement ratio?")
    assert r6["not_covered"] and not r6["sources"]            # graceful: no invented citation
    r7 = s.ask("Back to assignment - are there service charges?")
    assert r7["relation"] == "return_to_earlier" and r7["sources"][0]["doc"] == "Assignment Ashok.pdf"
    r8 = s.ask("Summarise what we covered")
    assert r8["relation"] == "meta"
    assert len(s.facts) >= 6 and s.turn == 8
    assert len(s.cited) >= 4
    # repeated question: assignee-details section was already cited, the prompt must flag it as already shown
    assert any(h["turn"] == 4 for h in s.history)
    print(s.transcript_md()[:1500])


def test_prompt_marks_already_shown_and_facts():
    seen = {}

    class Spy(FakeLLM):
        def json(self, prompt, *, tag=None, **kw):
            if tag[0] == "answer":
                seen[tag[1]] = prompt
            return super().json(prompt, tag=tag, **kw)

    s = SupportAssistant(KnowledgeBase(ROWS), Spy())
    s.ask("What is the assignment request form for?")
    s.ask("What details do I need for the assignee?")
    s.ask("Remind me what details the assignee needs?")
    assert "already shown to the user in turn 2" in seen[3]
    assert "fact from turn 2" in seen[3] and "fact from turn 1" in seen[3]
