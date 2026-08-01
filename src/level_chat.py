import streamlit as st
from src.rag_pipeline_hybrid import get_level_chain
from src.vector_store import get_vectorstore

LEVEL_ORDER = ["Junior", "Mid-Level", "Senior", "Architecture"]

LEVEL_META = {
    "Junior":       {"emoji": "🌱", "color": "#4CAF50", "desc": "Simple, foundational"},
    "Mid-Level":    {"emoji": "⚙️", "color": "#2196F3", "desc": "Concrete detail & decisions"},
    "Senior":       {"emoji": "🎯", "color": "#FF9800", "desc": "Tradeoffs & depth"},
    "Architecture": {"emoji": "🏛️", "color": "#9C27B0", "desc": "System-level design"},
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


def _default_pair(user_level):
    """Junior -> [Junior, Mid-Level], Mid-Level -> [Mid-Level, Senior],
    Senior -> [Senior, Architecture], Architecture -> [Architecture]."""
    if user_level not in LEVEL_ORDER:
        return LEVEL_ORDER[:2]
    idx = LEVEL_ORDER.index(user_level)
    if idx + 1 < len(LEVEL_ORDER):
        return [LEVEL_ORDER[idx], LEVEL_ORDER[idx + 1]]
    return [LEVEL_ORDER[idx]]


def display_level_chat():
    """One question, answered at the levels relevant to the user (with an override dropdown)."""

    st.title("🪜 Level Answers")
    st.markdown("See how the **same question** should be answered depending on the seniority you're interviewing for.")

    user_level = st.session_state.get("auth_profile", {}).get("level")
    default_selection = _default_pair(user_level)

    if user_level:
        st.caption(f"Your onboarding level: **{user_level}** — showing {' + '.join(default_selection)} by default.")

    selected_levels = st.multiselect(
        "Levels to show:",
        options=LEVEL_ORDER,
        default=default_selection,
        help="Pick any combination — e.g. select Senior + Architecture to preview those answers."
    )

    if "level_messages" not in st.session_state:
        st.session_state.level_messages = []

    for msg in st.session_state.level_messages:
        st.markdown(f"**{msg['role'].upper()}:** {msg['content']}")

    prompt = st.chat_input("Ask a question to see it answered at your selected levels...")
    if not prompt:
        return

    if not selected_levels:
        st.error("❌ Select at least one level above.")
        return

    vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
    if not vectorstore:
        st.warning("⚠️ Please go to Visual tab and process documents first.")
        return

    st.session_state.level_messages.append({"role": "user", "content": prompt})
    st.markdown(f"**USER:** {prompt}")

    answers = {}
    for i in range(0, len(selected_levels), 2):
        row = st.columns(min(2, len(selected_levels) - i), gap="medium")
        for slot, name in zip(row, selected_levels[i:i + 2]):
            with slot:
                with st.spinner(f"{LEVEL_META[name]['emoji']} Preparing {name} answer..."):
                    chain = get_level_chain(vectorstore, name)
                    answer = chain.invoke({"question": prompt})
                answers[name] = answer
                _level_card(name, answer)

    summary = "\n\n".join(f"[{lvl}] {ans}" for lvl, ans in answers.items())
    st.session_state.level_messages.append({"role": "assistant", "content": summary})
