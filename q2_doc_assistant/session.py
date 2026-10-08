"""Multi-turn, document-grounded support assistant.

Per turn (2 LLM calls):
  1. REWRITE  - resolve pronouns/ellipsis into a standalone search query and classify how this message relates
                to the conversation: follow_up | new_topic | return_to_earlier | meta | smalltalk
  2. ANSWER   - answer ONLY from retrieved sections, cite them, avoid repeating what was already told, and
                report the facts it gave (these are stored and shown back to the model next turn)

Session memory:
  history   verbatim user/assistant turns (also passed to the model)
  facts     what has already been told to the user, per turn, with the sections it came from
  cited     section -> first turn in which it was shown (so repeats can be recognised)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SYSTEM = (
    "You are a careful customer-support assistant for HDFC Life policy paperwork. You answer ONLY from the document "
    "excerpts you are given; you never use outside knowledge about insurance rules, products or HDFC Life. "
    "You reply with JSON only."
)

REWRITE_PROMPT = """Conversation so far (most recent last):
{history}

New user message: "{q}"

Return JSON:
{{"standalone_query": "<the message rewritten so it makes sense with no history: resolve pronouns and references; keep key terms>",
 "topic": "<3-5 word topic label>",
 "relation": "follow_up | new_topic | return_to_earlier | meta | smalltalk",
 "previous_topic": "<topic label of the previous turn or ''>"}}
Definitions: follow_up = continues the same topic; new_topic = unrelated to anything discussed; return_to_earlier = goes back to a
topic discussed before the previous turn; meta = asks about the conversation itself (what we covered, what I asked, summarise);
smalltalk = greeting/thanks."""

ANSWER_PROMPT = """DOCUMENT EXCERPTS (the only allowed source):
{excerpts}

KNOWLEDGE BASE TABLE OF CONTENTS (for suggesting related topics; do not cite from it):
{toc}

ALREADY TOLD TO THE USER IN THIS SESSION (do not repeat these):
{facts}

RECENT CONVERSATION:
{history}

Relation of the new message to the conversation: {relation} (previous topic: "{prev_topic}", current topic: "{topic}")
User's message: "{q}"

Rules:
1. Use only the excerpts. Put the excerpt tag right after each claim, e.g. "... must be signed by an appointee [S2]". Cite the narrowest excerpt.
2. An excerpt that merely shares words with the question does not answer it (e.g. a consent clause mentioning "claims settlement"
   does not answer a question about a settlement ratio). If the excerpts do not actually contain the answer, say plainly that these documents do not cover it, set "not_covered": true,
   cite nothing, and name 1-2 related topics from the table of contents that you CAN help with. Never guess.
3. Do not repeat information listed under ALREADY TOLD. If the user asks again, answer in one or two sentences pointing back
   ("As covered earlier, ...") and add only what is genuinely new from the excerpts, or offer a specific new angle.
   New relevant details from excerpts that were already shown are fine; re-stating old details is not.
4. relation = new_topic or return_to_earlier: begin with a short natural acknowledgement of the switch (e.g. "Switching to the
   proposal declarations -" / "Going back to the assignment form -"), then answer; do not drag in the old topic.
   relation = follow_up: build on the previous answer without re-introducing the topic.
5. relation = meta: answer from the conversation and the ALREADY TOLD list (no excerpts needed, citations optional).
   relation = smalltalk: reply in one friendly sentence and invite a question.
6. Plain language, concise (under ~120 words unless the user asks for detail). No headings. A short list is fine only for 3+ items.
Return JSON: {{"answer": "<text with [S#] tags>", "citations": ["S1"], "facts_given": ["<very short statement of each distinct thing you told the user>"], "not_covered": false}}"""


def _fmt_history(history, n=4, clip=350):
    if not history:
        return "(none)"
    out = []
    for h in history[-n:]:
        out.append(f"User (turn {h['turn']}): {h['user']}\nAssistant: {h['answer'][:clip]}{'...' if len(h['answer']) > clip else ''}")
    return "\n".join(out)


class SupportAssistant:
    def __init__(self, kb, llm, top_k=6, min_score=0.03):
        self.kb, self.llm, self.top_k, self.min_score = kb, llm, top_k, min_score
        self.history, self.facts, self.cited = [], [], {}
        self.turn = 0

    # -------------------------------------------------------------- one turn
    def ask(self, q: str) -> dict:
        self.turn += 1
        rw = self.llm.json(REWRITE_PROMPT.format(history=_fmt_history(self.history), q=q), system=SYSTEM, temperature=0.0,
                           tag=("rewrite", self.turn))
        relation = rw.get("relation", "follow_up")
        if relation not in {"follow_up", "new_topic", "return_to_earlier", "meta", "smalltalk"}:
            relation = "follow_up"
        if not self.history and relation in {"follow_up", "return_to_earlier"}:
            relation = "new_topic"
        topic = rw.get("topic") or "general"
        query = rw.get("standalone_query") or q

        hits = [] if relation in {"meta", "smalltalk"} else self.kb.search(query, k=self.top_k, min_score=self.min_score)
        if relation == "follow_up" and self.history:  # "why?" style follow-ups keep the previous evidence in play
            have = {c.id for c, _ in hits}
            for cid in self.history[-1]["chunk_ids"]:
                if cid not in have and len(hits) < self.top_k + 2:
                    hits.append((self.kb.chunks[cid], 0.0))
        tags = {f"S{i + 1}": c for i, (c, _) in enumerate(hits)}

        ex = []
        for t, c in tags.items():
            seen = f" (already shown to the user in turn {self.cited[c.id]})" if c.id in self.cited else ""
            ex.append(f"[{t}] {c.label}{seen}\n{c.text}")
        facts = "\n".join(f"- (turn {f['turn']}) {f['fact']}" for f in self.facts) or "(nothing yet)"
        out = self.llm.json(ANSWER_PROMPT.format(
            excerpts="\n\n".join(ex) or "(no relevant excerpts found)", toc=self.kb.toc(), facts=facts,
            history=_fmt_history(self.history, n=6, clip=500), relation=relation,
            prev_topic=rw.get("previous_topic") or (self.history[-1]["topic"] if self.history else ""),
            topic=topic, q=q), system=SYSTEM, temperature=0.2, tag=("answer", self.turn))

        text = (out.get("answer") or "").strip()
        not_covered = bool(out.get("not_covered"))
        used = [t for t in re.findall(r"\[(S\d+)\]", text) if t in tags]
        used += [t for t in out.get("citations", []) if t in tags and t not in used]
        order = list(dict.fromkeys(used))
        if not order and not not_covered and relation not in {"meta", "smalltalk"} and hits:
            order = ["S1"]  # model forgot to cite: show the closest section rather than nothing
            text += " [S1]"
        num = {t: i + 1 for i, t in enumerate(order)}
        text = re.sub(r"\[(S\d+)\]", lambda m: f"[{num[m.group(1)]}]" if m.group(1) in num else "", text)
        text = re.sub(r"\s+([.,;])", r"\1", re.sub(r" {2,}", " ", text)).strip()
        sources = [dict(n=num[t], label=tags[t].label, doc=tags[t].doc, page=tags[t].page, section=tags[t].section) for t in order]

        for t in order:
            self.cited.setdefault(tags[t].id, self.turn)
        for fact in out.get("facts_given", []) or []:
            if fact:
                self.facts.append(dict(turn=self.turn, fact=str(fact), sources=[tags[t].label for t in order]))
        rec = dict(turn=self.turn, user=q, answer=text, topic=topic, relation=relation, query=query,
                   sources=sources, chunk_ids=[tags[t].id for t in order], not_covered=not_covered)
        self.history.append(rec)
        return rec

    # -------------------------------------------------------------- transcript
    def transcript_md(self, title="Support assistant - demo transcript") -> str:
        lines = [f"# {title}", ""]
        for h in self.history:
            lines += [f"### Turn {h['turn']}  _(relation: {h['relation']} · topic: {h['topic']})_", f"**User:** {h['user']}", "",
                      f"**Assistant:** {h['answer']}", ""]
            if h["sources"]:
                lines += ["Sources:"] + [f"- [{s['n']}] {s['label']}" for s in h["sources"]] + [""]
            elif h["not_covered"]:
                lines += ["_Not covered by the documents - no citation._", ""]
        return "\n".join(lines)
