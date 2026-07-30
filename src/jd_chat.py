import streamlit as st
from src.rag_pipeline_hybrid import generate_jd_questions, CATEGORY_DEFINITIONS
from src.vector_store import get_vectorstore

CATEGORY_STYLE = {
    "Technical": ("🛠️", "#2196F3"),
    "Behavioral": ("🗣️", "#4CAF50"),
    "Resume": ("📄", "#9C27B0"),
    "Gap": ("⚠️", "#FF5722"),
    "General": ("📌", "#607D8B"),
}


def _qa_card(item):
    emoji, color = CATEGORY_STYLE.get(item["category"], CATEGORY_STYLE["General"])
    with st.container(border=True):
        st.markdown(f"""
        <span style="background:{color}; color:white; padding:2px 10px; border-radius:12px;
                     font-size:12px; font-weight:600;">{emoji} {item['category']}</span>
        """, unsafe_allow_html=True)
        st.markdown(f"**Q: {item['question']}**")
        st.markdown(f"A: {item['answer']}")


def display_jd_prep():
    """Paste a JD, get a full set of likely interview questions + resume-grounded answers."""

    st.title("📋 JD Answers")
    st.markdown("Paste a job description — get the questions you're likely to be asked, with resume-backed answers ready to go.")
    st.divider()

    jd_text = st.text_area("Paste the job description here:", height=200, placeholder="Paste JD text...")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_categories = st.multiselect(
            "What type of questions do you want?",
            options=list(CATEGORY_DEFINITIONS.keys()),
            default=list(CATEGORY_DEFINITIONS.keys()),
            help="Resume = questions about your own project/role/responsibilities, answered straight from your resume."
        )
    with col_b:
        num_questions = st.slider("How many", min_value=4, max_value=15, value=8)

    if st.button("🎯 Generate Interview Prep", use_container_width=True):
        if not jd_text.strip():
            st.error("❌ Please paste a job description first.")
            return
        if not selected_categories:
            st.error("❌ Pick at least one question type.")
            return

        vectorstore = st.session_state.get("vectorstore") or get_vectorstore()
        if not vectorstore:
            st.warning("⚠️ Please go to Visual tab and process documents first.")
            return

        with st.spinner("🔍 Matching your resume against this JD and generating questions..."):
            items = generate_jd_questions(vectorstore, jd_text, num_questions, categories=selected_categories)

        if not items:
            st.error("❌ Couldn't generate questions — try again.")
            return

        st.session_state.jd_items = items
        st.session_state.jd_text = jd_text
        st.session_state.jd_categories = selected_categories

    if "jd_items" in st.session_state:
        items = st.session_state.jd_items
        st.success(f"✅ {len(items)} questions ready")

        counts = {}
        for i in items:
            counts[i["category"]] = counts.get(i["category"], 0) + 1
        cols = st.columns(len(counts) or 1)
        for col, (cat, count) in zip(cols, counts.items()):
            emoji, _ = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["General"])
            col.metric(f"{emoji} {cat}", count)

        st.divider()
        for item in items:
            _qa_card(item)

        st.divider()
        if st.button("➕ More questions", use_container_width=True):
            vectorstore = st.session_state.get("vectorstore") or get_vectorstore()
            if not vectorstore:
                st.warning("⚠️ Please go to Visual tab and process documents first.")
                return
            with st.spinner("🔍 Generating more questions (no repeats)..."):
                existing_questions = [i["question"] for i in st.session_state.jd_items]
                more_items = generate_jd_questions(
                    vectorstore,
                    st.session_state.jd_text,
                    num_questions=5,
                    categories=st.session_state.jd_categories,
                    exclude_questions=existing_questions
                )
            st.session_state.jd_items = st.session_state.jd_items + more_items
            st.rerun()
