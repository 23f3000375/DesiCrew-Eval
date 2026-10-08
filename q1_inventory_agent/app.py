"""Streamlit chat UI.   streamlit run q1_inventory_agent/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q1_inventory_agent.agent import InventoryAgent  # noqa: E402

st.set_page_config(page_title="Inventory Analyst Agent", page_icon="📦", layout="wide")
st.title("📦 Inventory Analyst Agent")
st.caption("Ask questions about the inventory sheet. The agent writes and runs pandas code, searches the web for "
           "definitions, and explains the result in plain English.")

SAMPLES = [
    "Which 5 products hold the most inventory value, and what share of the total is that?",
    "Which products sold the highest percentage of the stock they had available?",
    "Do the hand-in-stock numbers reconcile with opening + purchased - sold? Show any that don't.",
    "What is inventory turnover? Calculate it for the whole catalog using the sheet.",
    "Chart units sold for the top 10 products.",
]

if "agent" not in st.session_state:
    try:
        st.session_state.agent = InventoryAgent()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not start the agent: {e}")
        st.stop()
    st.session_state.messages = []

agent: InventoryAgent = st.session_state.agent

with st.sidebar:
    st.subheader("Dataset")
    st.dataframe(agent.df, height=300, width='stretch')
    st.caption(f"{len(agent.df)} products")
    st.subheader("Try asking")
    picked = None
    for i, s in enumerate(SAMPLES):
        if st.button(s, key=f"s{i}", width='stretch'):
            picked = s
    if st.button("🔄 New conversation", width='stretch'):
        agent.reset()
        st.session_state.messages = []
        st.rerun()


def render_steps(steps):
    for s in steps:
        if s["kind"] == "code":
            with st.expander("🧮 Code the agent ran", expanded=False):
                st.code(s["input"], language="python")
                st.text(s["error"] or s["output"] or "(no output)")
        else:
            with st.expander(f"🔎 Web search: {s['input']}", expanded=False):
                st.write(s["output"])
                for src in s["sources"][:5]:
                    st.caption(src)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m["role"] == "assistant":
            for c in m.get("charts", []):
                st.image(c)
            render_steps(m.get("steps", []))
            if m.get("model"):
                st.caption(f"answered by {m['model']}")

prompt = st.chat_input("Ask about the inventory…") or picked
if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                r = agent.ask(prompt)
                steps = [dict(kind=s.kind, input=s.input, output=s.output, error=s.error, sources=s.sources) for s in r.steps]
                msg = dict(role="assistant", text=r.text, charts=r.charts, steps=steps, model=agent.llm.model)
            except Exception as e:  # noqa: BLE001
                msg = dict(role="assistant", text=f"Sorry, something went wrong: {e}", charts=[], steps=[])
        st.markdown(msg["text"])
        for c in msg["charts"]:
            st.image(c)
        render_steps(msg["steps"])
        if msg.get("model"):
            st.caption(f"answered by {msg['model']}")
    st.session_state.messages.append(msg)
