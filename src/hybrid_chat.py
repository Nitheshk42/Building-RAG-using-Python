import streamlit as st
from src.rag_pipeline_hybrid import route_question, get_resume_chain, get_technical_chain
from src.vector_store import get_vectorstore

ROUTE_COLORS = {
    "RESUME_FACT": "#4CAF50",
    "TECHNICAL_DEEP_DIVE": "#FF9800",
    "BOTH": "#2196F3",
}


def _side_header(title, emoji, color, subtitle):
    st.markdown(f"""
    <div style="border-left: 6px solid {color}; padding: 10px 14px;
                border-radius: 6px; background: rgba(128,128,128,0.06); margin-bottom: 8px;">
        <span style="font-size: 19px; font-weight: 700; color: {color};">{emoji} {title}</span>
        <div style="font-size: 12px; opacity: 0.7; margin-top: -2px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def display_hybrid_chat():
    """Hybrid Chat: Resume Fact vs Technical Deep-Dive, side-by-side"""

    st.title("🔀 Hybrid Chat")
    st.markdown("Real routing decision + two independently generated answers, so you always have an interview-ready technical answer alongside the resume-grounded one.")
    st.divider()

    if "hybrid_messages" not in st.session_state:
        st.session_state.hybrid_messages = []

    for msg in st.session_state.hybrid_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your experience or technical concepts..."):
        st.session_state.hybrid_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))
        if not vectorstore:
            st.warning("⚠️ Please go to Visual tab and process documents first.")
            return

        with st.spinner("🧭 Routing question..."):
            category, reason = route_question(prompt)

        color = ROUTE_COLORS.get(category, "#607D8B")
        st.markdown(f"""
        <div style="border: 1px solid {color}; border-radius: 8px; padding: 10px 14px; margin: 10px 0;">
            🧭 <b style="color:{color};">Routing decision: {category}</b><br>
            <span style="opacity:0.8;">{reason}</span>
        </div>
        """, unsafe_allow_html=True)

        col_resume, col_technical = st.columns(2, gap="large")

        with col_resume:
            _side_header("From Your Resume", "📄", "#4CAF50", "Strictly grounded in retrieved chunks")
            with st.spinner("Retrieving resume facts..."):
                resume_chain = get_resume_chain(vectorstore)
                resume_response = resume_chain.invoke({"question": prompt})
            with st.container(border=True):
                st.markdown(resume_response)

        with col_technical:
            _side_header("Technical Deep-Dive", "🧠", "#FF9800", "Interview follow-up style: approach, challenges, resolution")
            with st.spinner("Generating interview-style answer..."):
                technical_chain = get_technical_chain(vectorstore)
                technical_response = technical_chain.invoke({"question": prompt})
            with st.container(border=True):
                st.markdown(technical_response)

        st.session_state.hybrid_messages.append({
            "role": "assistant",
            "content": f"[{category}] Resume: {resume_response}\n\nTechnical: {technical_response}"
        })
