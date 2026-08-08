import streamlit as st
from src.rag_pipeline_hybrid import generate_general_jd_questions

LEVEL_META = {
    "Junior":    {"emoji": "🌱", "color": "#4CAF50"},
    "Mid-Level": {"emoji": "⚙️", "color": "#2196F3"},
    "Senior":    {"emoji": "🎯", "color": "#FF9800"},
    "Architect": {"emoji": "🏛️", "color": "#9C27B0"},
}
LEVEL_ORDER = ["Junior", "Mid-Level", "Senior", "Architect"]


def _hero():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6c5ce7 0%, #1a73e8 100%);
                padding: 22px 26px; border-radius: 14px; margin-bottom: 18px;">
        <div style="font-size: 24px; font-weight: 700; color: white;">🧠 General JD Answers</div>
        <div style="font-size: 14px; color: rgba(255,255,255,0.9); margin-top: 4px;">
            Paste a job description — get interview Q&amp;A generated purely from general
            domain knowledge, the way a general-purpose AI would answer with just the JD and
            nothing else. <b>No resume, no personal context</b> — pure LLM knowledge, at all
            four seniority levels.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _level_answer_block(question_idx, item):
    st.markdown(f"**Q{question_idx}. {item['question']}**")
    for i in range(0, len(LEVEL_ORDER), 2):
        row = st.columns(2, gap="medium")
        for slot, level in zip(row, LEVEL_ORDER[i:i + 2]):
            meta = LEVEL_META[level]
            answer = item["answers"].get(level, "").strip()
            with slot:
                st.markdown(f"""
                <div style="border-left: 5px solid {meta['color']}; padding: 6px 12px;
                            border-radius: 6px; background: rgba(128,128,128,0.06); margin-bottom: 4px;">
                    <span style="font-weight:700; color:{meta['color']};">{meta['emoji']} {level}</span>
                </div>
                """, unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(answer if answer else "_No answer generated for this level._")
    st.divider()


def display_general_jd_prep():
    """Paste a JD, get interview Q&A generated purely from general LLM knowledge - no resume
    grounding at all, at all four seniority levels. This is the 'no personal context' counterpart
    to the resume-grounded My JD Answers tab."""

    _hero()

    with st.container(border=True):
        st.markdown("**Paste the job description**")
        jd_text = st.text_area(
            "Job description",
            height=160,
            placeholder="Paste the job description here...",
            label_visibility="collapsed",
            key="general_jd_text_input"
        )
        generate_clicked = st.button("🧠 Generate General Q&A", use_container_width=True, type="primary")

    if generate_clicked:
        if not jd_text.strip():
            st.error("❌ Please paste a job description first.")
            return
        with st.spinner("🧠 Generating questions and level-by-level answers from general knowledge..."):
            items = generate_general_jd_questions(jd_text, num_questions=6)
        if not items:
            st.error("❌ Couldn't generate questions — try again.")
            return
        st.session_state.general_jd_items = items
        st.session_state.general_jd_text = jd_text

    if "general_jd_items" not in st.session_state:
        return

    items = st.session_state.general_jd_items
    st.write("")
    st.success(f"✅ {len(items)} question(s) generated — no resume context used.")
    st.write("")

    for idx, item in enumerate(items, 1):
        _level_answer_block(idx, item)

    if st.button("➕ 5 more questions", use_container_width=True):
        with st.spinner("🧠 Generating 5 more questions (no repeats)..."):
            existing_questions = [i["question"] for i in st.session_state.general_jd_items]
            more_items = generate_general_jd_questions(
                st.session_state.general_jd_text,
                num_questions=5,
                exclude_questions=existing_questions
            )
        st.session_state.general_jd_items = st.session_state.general_jd_items + more_items
        st.rerun()
