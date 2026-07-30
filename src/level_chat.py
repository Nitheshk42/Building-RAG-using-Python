import streamlit as st
from src.rag_pipeline_hybrid import get_level_chain
from src.vector_store import get_vectorstore

LEVELS = [
    {"name": "Junior",       "emoji": "🌱", "color": "#4CAF50", "desc": "Simple, foundational"},
    {"name": "Mid-Level",    "emoji": "⚙️", "color": "#2196F3", "desc": "Concrete detail & decisions"},
    {"name": "Senior",       "emoji": "🎯", "color": "#FF9800", "desc": "Tradeoffs & depth"},
    {"name": "Architecture", "emoji": "🏛️", "color": "#9C27B0", "desc": "System-level design"},
]


def _level_card(level, answer):
    st.markdown(f"""
    <div style="border-left: 6px solid {level['color']}; padding: 10px 14px;
                border-radius: 6px; background: rgba(128,128,128,0.06); margin-bottom: 8px;">
        <span style="font-size: 18px; font-weight: 700; color: {level['color']};">
            {level['emoji']} {level['name']}
        </span>
        <div style="font-size: 12px; opacity: 0.7; margin-top: -2px;">{level['desc']}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(answer)


def display_level_chat():
    """One question, answered at 4 seniority levels side-by-side."""

    st.title("🪜 Level Answers")
    st.markdown("See how the **same question** should be answered depending on the seniority you're interviewing for.")

    if "level_messages" not in st.session_state:
        st.session_state.level_messages = []

    for msg in st.session_state.level_messages:
        st.markdown(f"**{msg['role'].upper()}:** {msg['content']}")

    prompt = st.chat_input("Ask a question to see it answered at every level...")
    if not prompt:
        return

    vectorstore = st.session_state.get("vectorstore") or get_vectorstore()
    if not vectorstore:
        st.warning("⚠️ Please go to Visual tab and process documents first.")
        return

    st.session_state.level_messages.append({"role": "user", "content": prompt})
    st.markdown(f"**USER:** {prompt}")

    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")
    slots = [row1[0], row1[1], row2[0], row2[1]]

    answers = {}
    for slot, level in zip(slots, LEVELS):
        with slot:
            with st.spinner(f"{level['emoji']} Preparing {level['name']} answer..."):
                chain = get_level_chain(vectorstore, level["name"])
                answer = chain.invoke({"question": prompt})
            answers[level["name"]] = answer
            _level_card(level, answer)

    summary = "\n\n".join(f"[{lvl}] {ans}" for lvl, ans in answers.items())
    st.session_state.level_messages.append({"role": "assistant", "content": summary})
