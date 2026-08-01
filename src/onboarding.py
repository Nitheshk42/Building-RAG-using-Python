import streamlit as st
import os
import shutil
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.auth import set_profile_level, mark_resume_uploaded

LEVELS = ["Junior", "Mid-Level", "Senior", "Architecture"]


def user_data_dir(username):
    safe_name = username.replace("/", "_").replace("\\", "_")
    return os.path.join("data", safe_name)


def process_resume(uploaded_files, username):
    """Saves uploaded resume file(s), wipes any previous resume for this user,
    and reprocesses into a fresh vector store. Used by both onboarding and the
    'upload a different resume' flow — no dependency on the Visual tab."""
    data_dir = user_data_dir(username)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)  # clear previous resume for this user
    os.makedirs(data_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    documents = load_documents(data_dir)
    splits = split_documents(documents)
    embeddings_model = get_embeddings()
    vectorstore = create_vector_store(splits, embeddings_model, username=username)

    st.session_state.splits = splits
    st.session_state.documents = documents
    st.session_state.embeddings_model = embeddings_model
    st.session_state.vectorstore = vectorstore
    mark_resume_uploaded(username)


def display_onboarding():
    """Runs once after login: upload resume, pick target level, process into the vector DB."""
    st.title("👋 Welcome to StudySage")
    st.markdown("Let's get you set up — upload your resume and pick the level you're interviewing for.")
    st.divider()

    st.subheader("1️⃣ Upload your resume (PDF or Word)")
    st.caption("One file only — keeps answers grounded in a single, consistent resume.")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "doc"], accept_multiple_files=False
    )
    uploaded_files = [uploaded_file] if uploaded_file else []

    st.subheader("2️⃣ What level are you interviewing for?")
    level = st.radio("Target level:", LEVELS, horizontal=True)

    if st.button("🚀 Get Started", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("❌ Please upload at least one resume file.")
            return

        username = st.session_state.auth_user
        with st.spinner("📄 Saving and processing your resume..."):
            process_resume(uploaded_files, username)
        set_profile_level(username, level)
        st.session_state.auth_profile = {"level": level, "resume_uploaded": True}

        st.success("✅ All set! Taking you to StudySage...")
        st.rerun()
