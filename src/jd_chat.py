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


def _hero():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 100%);
                padding: 22px 26px; border-radius: 14px; margin-bottom: 18px;">
        <div style="font-size: 24px; font-weight: 700; color: white;">📋 JD Answers</div>
        <div style="font-size: 14px; color: rgba(255,255,255,0.9); margin-top: 4px;">
            Paste a job description — get the questions you're likely to be asked, matched
            against your actual resume, with answers ready to go.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _qa_card(item, index):
    emoji, color = CATEGORY_STYLE.get(item["category"], CATEGORY_STYLE["General"])
    st.markdown(f"""
    <div style="border-left: 4px solid {color}; border-radius: 10px; padding: 16px 18px;
                margin-bottom: 12px; background: rgba(255,255,255,0.03);
                box-shadow: 0 1px 3px rgba(0,0,0,0.25);">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span style="background:{color}; color:white; padding:2px 10px; border-radius:10px;
                         font-size:11px; font-weight:700; letter-spacing:0.3px;">{emoji} {item['category'].upper()}</span>
            <span style="opacity:0.35; font-size:12px;">Q{index}</span>
        </div>
        <div style="font-size:16px; font-weight:600; margin-bottom:8px;">{item['question']}</div>
        <div style="font-size:14.5px; line-height:1.6; opacity:0.92;">{item['answer']}</div>
    </div>
    """, unsafe_allow_html=True)


def display_jd_prep():
    """Paste a JD, get a full set of likely interview questions + resume-grounded answers."""

    _hero()

    with st.container(border=True):
        st.markdown("**Paste the job description**")
        jd_text = st.text_area(
            "Job description",
            height=160,
            placeholder="Paste the job description here...",
            label_visibility="collapsed"
        )
        generate_clicked = st.button("🎯 Generate Interview Prep", use_container_width=True, type="primary")

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
                vectorstore, jd_text, num_questions=5, categories=list(CATEGORY_DEFINITIONS.keys())
            )

        if not items:
            st.error("❌ Couldn't generate questions — try again.")
            return

        st.session_state.jd_items = items
        st.session_state.jd_text = jd_text

    if "jd_items" not in st.session_state:
        return

    items = st.session_state.jd_items
    st.write("")

    counts = {}
    for i in items:
        counts[i["category"]] = counts.get(i["category"], 0) + 1

    summary_cols = st.columns(len(counts) + 1)
    summary_cols[0].metric("✅ Total", len(items))
    for col, (cat, count) in zip(summary_cols[1:], counts.items()):
        emoji, _ = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["General"])
        col.metric(f"{emoji} {cat}", count)

    st.write("")

    present_categories = [c for c in CATEGORY_DEFINITIONS.keys() if any(i["category"] == c for i in items)]
    category_tabs = st.tabs([f"{CATEGORY_STYLE.get(c, CATEGORY_STYLE['General'])[0]} {c} ({counts[c]})" for c in present_categories])

    for tab, category in zip(category_tabs, present_categories):
        with tab:
            cat_items = [i for i in items if i["category"] == category]
            for idx, item in enumerate(cat_items, 1):
                _qa_card(item, idx)

    st.write("")
    if st.button("➕ 5 more questions", use_container_width=True):
        vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
        if not vectorstore:
            st.warning("⚠️ Please go to Visual tab and process documents first.")
            return
        with st.spinner("🔍 Generating 5 more questions (no repeats)..."):
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
