import streamlit as st
from src.rag_pipeline_hybrid import get_level_chain
from src.vector_store import get_vectorstore

LEVEL_ORDER = ["Junior", "Mid-Level", "Senior", "Architect"]

LEVEL_META = {
    "Junior":       {"emoji": "🌱", "color": "#4CAF50", "desc": "Simple, foundational"},
    "Mid-Level":    {"emoji": "⚙️", "color": "#2196F3", "desc": "Concrete detail & decisions"},
    "Senior":       {"emoji": "🎯", "color": "#FF9800", "desc": "Tradeoffs & depth"},
    "Architect": {"emoji": "🏛️", "color": "#9C27B0", "desc": "System-level design"},
}


def _level_card(name, answer):
    meta = LEVEL_META[name]
    st.markdown(f"""
    <div style="border-left: 6px solid {meta['color']}; padding: 10px 14px;
                border-radius: 6px; background: rgba(128,128,128,0.06); margin-bottom: 8px;">
        <span style="font-size: 18px; font-weight: 700; color: {meta['color']};">
            {meta['emoji']} {name}
        </span>
        <div style="font-size: 12px; opacity: 0.7; margin-top: -2px;">{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(answer)


def display_level_chat():
    """One question, answered at all 4 seniority levels side-by-side."""

    st.title("🪜 Level Answers")
    st.markdown("See how the **same question** should be answered depending on the seniority you're interviewing for.")

    if "level_messages" not in st.session_state:
        st.session_state.level_messages = []

    for msg in st.session_state.level_messages:
        st.markdown(f"**{msg['role'].upper()}:** {msg['content']}")

    prompt = st.chat_input("Ask a question to see it answered at every level...")
    if not prompt:
        return

    vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
    if not vectorstore:
        st.warning("⚠️ Please go to Visual tab and process documents first.")
        return

    st.session_state.level_messages.append({"role": "user", "content": prompt})
    st.markdown(f"**USER:** {prompt}")

    answers = {}
    for i in range(0, len(LEVEL_ORDER), 2):
        row = st.columns(2, gap="medium")
        for slot, name in zip(row, LEVEL_ORDER[i:i + 2]):
            with slot:
                with st.spinner(f"{LEVEL_META[name]['emoji']} Preparing {name} answer..."):
                    chain = get_level_chain(vectorstore, name)
                    answer = chain.invoke({"question": prompt})
                answers[name] = answer
                _level_card(name, answer)

    summary = "\n\n".join(f"[{lvl}] {ans}" for lvl, ans in answers.items())
    st.session_state.level_messages.append({"role": "assistant", "content": summary})
