"""Streamlit chat UI.   streamlit run q2_doc_assistant/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import LLM  # noqa: E402
from kb import KnowledgeBase, ingest  # noqa: E402
from session import SupportAssistant  # noqa: E402

st.set_page_config(page_title="HDFC Life Document Assistant", page_icon="📄", layout="wide")
st.title("📄 Document-aware Support Assistant")
st.caption("Answers come only from the ingested HDFC Life forms, with the document, page and section cited. "
           "It remembers this conversation and avoids repeating itself.")

if "assistant" not in st.session_state:
    try:
        llm = LLM()
        with st.spinner("Loading knowledge base (first run transcribes the scanned PDFs with Gemini vision)…"):
            kb = KnowledgeBase(ingest(llm, mode="vision"))
        st.session_state.assistant = SupportAssistant(kb, llm)
        st.session_state.llm, st.session_state.kb = llm, kb
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not start: {e}")
        st.stop()

bot: SupportAssistant = st.session_state.assistant

with st.sidebar:
    st.subheader("Knowledge base")
    st.text(bot.kb.toc())
    st.caption(f"{len(bot.kb.chunks)} sections indexed")
    st.subheader("Session memory")
    if bot.history:
        st.markdown("**Topic trail:** " + " → ".join(h["topic"] for h in bot.history))
        with st.expander(f"Facts already given ({len(bot.facts)})"):
            for f in bot.facts:
                st.markdown(f"- _turn {f['turn']}_: {f['fact']}")
    else:
        st.caption("Nothing yet.")
    if st.button("🔄 New session", width='stretch'):
        st.session_state.assistant = SupportAssistant(bot.kb, st.session_state.llm)
        st.rerun()

for h in bot.history:
    with st.chat_message("user"):
        st.markdown(h["user"])
    with st.chat_message("assistant"):
        st.markdown(h["answer"])
        for s in h["sources"]:
            st.caption(f"[{s['n']}] {s['label']}")
        if h["not_covered"]:
            st.caption("Not covered by the documents.")

if q := st.chat_input("Ask about the assignment form or the proposal declarations…"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                r = bot.ask(q)
            except Exception as e:  # noqa: BLE001
                bot.turn -= 1
                st.error(f"Something went wrong: {e}")
                st.stop()
    st.rerun()
