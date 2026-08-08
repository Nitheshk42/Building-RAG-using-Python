import streamlit as st
import os
import shutil
from src.auth import set_profile_level, mark_resume_uploaded

# NOTE: document_loader / text_splitter / embeddings / vector_store are imported lazily
# inside process_resume() below, not here. Those pull in torch, sentence-transformers,
# and chromadb - a genuinely heavy import chain. Importing onboarding.py just to show the
# "upload your resume" screen was forcing that entire chain to load before the process
# could even bind to $PORT, which is what was causing connection timeouts on cold starts.
# Deferring them means the app boots instantly and only pays that cost once, at the moment
# a resume is actually being processed.

LEVELS = ["Junior", "Mid-Level", "Senior", "Architect"]

# Same reasoning as auth.py/vector_store.py: uploaded resume files need to live on a
# persistent mount in production (Cloud Run's local disk doesn't survive across instances).
DATA_ROOT = os.getenv("APP_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_data_dir(username):
    safe_name = username.replace("/", "_").replace("\\", "_")
    return os.path.join(DATA_ROOT, "data", safe_name)


def process_resume(uploaded_files, username):
    """Saves uploaded resume file(s), wipes any previous resume for this user,
    and reprocesses into a fresh vector store. Used by both onboarding and the
    'upload a different resume' flow — no dependency on the Visual tab."""
    if st.session_state.get("vectorstore_creating"):
        # Reentrancy guard: Streamlit can occasionally rerun a script mid-execution (another
        # widget's state changing while this is still running), which can abort a chromadb
        # write partway through and leave its SQLite file in a bad state ("attempt to write a
        # readonly database" on the next attempt). Refusing a second concurrent call for this
        # session avoids that race.
        st.warning("⏳ Already processing a resume — please wait for it to finish.")
        return
    st.session_state.vectorstore_creating = True
    try:
        data_dir = user_data_dir(username)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)  # clear previous resume for this user
        os.makedirs(data_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        from src.document_loader import load_documents
        from src.text_splitter import split_documents
        from src.embeddings import get_embeddings
        from src.vector_store import create_vector_store

        documents = load_documents(data_dir)
        splits = split_documents(documents)
        embeddings_model = get_embeddings()
        vectorstore = create_vector_store(splits, embeddings_model, username=username)

        st.session_state.splits = splits
        st.session_state.documents = documents
        st.session_state.embeddings_model = embeddings_model
        st.session_state.vectorstore = vectorstore
        mark_resume_uploaded(username)
    finally:
        st.session_state.vectorstore_creating = False


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
