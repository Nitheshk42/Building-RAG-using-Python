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


def _qa_card(item, index):
    emoji, color = CATEGORY_STYLE.get(item["category"], CATEGORY_STYLE["General"])
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <span style="background:{color}; color:white; padding:3px 12px; border-radius:12px;
                         font-size:12px; font-weight:600;">{emoji} {item['category']}</span>
            <span style="opacity:0.4; font-size:12px;">#{index}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"##### {item['question']}")
        st.markdown(item["answer"])


def display_jd_prep():
    """Paste a JD, get a full set of likely interview questions + resume-grounded answers."""

    st.title("📋 JD Answers")
    st.markdown(
        "Paste a job description below — this pulls your most relevant resume experience "
        "and generates the questions you're likely to be asked, with answers ready to go."
    )
    st.divider()

    with st.container(border=True):
        jd_text = st.text_area(
            "Job description",
            height=180,
            placeholder="Paste the job description here...",
            label_visibility="collapsed"
        )
        col_slider, col_btn = st.columns([3, 1])
        with col_slider:
            num_questions = st.slider("Number of questions", min_value=4, max_value=15, value=8)
        with col_btn:
            st.write("")
            generate_clicked = st.button("🎯 Generate", use_container_width=True, type="primary")

    if generate_clicked:
        if not jd_text.strip():
            st.error("❌ Please paste a job description first.")
            return

        vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
        if not vectorstore:
            st.warning("⚠️ Please go to Visual tab and process documents first.")
            return

        with st.spinner("🔍 Matching your resume against this JD and generating questions..."):
            items = generate_jd_questions(
                vectorstore, jd_text, num_questions, categories=list(CATEGORY_DEFINITIONS.keys())
            )

        if not items:
            st.error("❌ Couldn't generate questions — try again.")
            return

        st.session_state.jd_items = items
        st.session_state.jd_text = jd_text

    if "jd_items" not in st.session_state:
        return

    items = st.session_state.jd_items
    st.divider()
    st.success(f"✅ {len(items)} questions ready — grouped by category below")

    counts = {}
    for i in items:
        counts[i["category"]] = counts.get(i["category"], 0) + 1
    cols = st.columns(len(counts) or 1)
    for col, (cat, count) in zip(cols, counts.items()):
        emoji, _ = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["General"])
        col.metric(f"{emoji} {cat}", count)

    st.divider()

    for category in CATEGORY_DEFINITIONS.keys():
        cat_items = [i for i in items if i["category"] == category]
        if not cat_items:
            continue
        emoji, color = CATEGORY_STYLE.get(category, CATEGORY_STYLE["General"])
        st.markdown(f"### {emoji} {category}")
        for idx, item in enumerate(cat_items, 1):
            _qa_card(item, idx)
        st.write("")

    st.divider()
    if st.button("➕ More questions", use_container_width=True):
        vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
        if not vectorstore:
            st.warning("⚠️ Please go to Visual tab and process documents first.")
            return
        with st.spinner("🔍 Generating more questions (no repeats)..."):
            existing_questions = [i["question"] for i in st.session_state.jd_items]
            more_items = generate_jd_questions(
                vectorstore,
                st.session_state.jd_text,
                num_questions=5,
                categories=list(CATEGORY_DEFINITIONS.keys()),
                exclude_questions=existing_questions
            )
        st.session_state.jd_items = st.session_state.jd_items + more_items
        st.rerun()
