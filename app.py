__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from src.visual_pipeline import display_visual_pipeline
from src.chat import display_chat_interface
from src.hybrid_chat import display_hybrid_chat
from src.level_chat import display_level_chat
from src.jd_chat import display_jd_prep
from src.auth_ui import display_auth
from src.onboarding import display_onboarding, process_resume
from src.auth import get_profile

st.set_page_config(page_title="StudySage", page_icon="📚", layout="wide")

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ===== GATE 1: LOGIN / SIGNUP =====
if not st.session_state.auth_user:
    display_auth()
    st.stop()

# ===== GATE 2: ONBOARDING (upload resume + pick level) =====
profile = st.session_state.get("auth_profile") or get_profile(st.session_state.auth_user)
st.session_state.auth_profile = profile

if not profile.get("resume_uploaded"):
    display_onboarding()
    st.stop()


@st.dialog("Reprocess with this resume?")
def _confirm_reprocess_dialog():
    st.write(
        "Uploading a new resume will **wipe your current resume's data** and replace it "
        "with this one."
    )
    st.caption(
        "We do this on purpose: keeping only one resume active at a time means every answer "
        "stays precise and grounded in a single, consistent source. If both resumes stayed in "
        "the same search index, their content could blend together and answers might mix up "
        "details from both — which isn't something you want to say to a recruiter or vendor."
    )
    st.write("You'll be able to use the new resume right away — no need to visit the Visual RAG tab.")
    col_yes, col_cancel = st.columns(2)
    with col_yes:
        if st.button("🔄 Yes, reprocess", use_container_width=True, type="primary"):
            with st.spinner("📄 Reprocessing your new resume..."):
                process_resume(st.session_state.pending_new_resume, st.session_state.auth_user)
            st.session_state.pop("pending_new_resume", None)
            st.success("✅ New resume is now active!")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_new_resume", None)
            st.rerun()


# ===== MAIN APP =====
with st.sidebar:
    st.markdown(f"👤 **{st.session_state.auth_user}**")
    st.caption(f"Level: {profile.get('level', 'Not set')}")

    with st.expander("📄 Upload a different resume"):
        st.caption("One file only.")
        new_file = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "doc"], accept_multiple_files=False, key="resume_reupload"
        )
        if new_file and st.button("Use this resume", use_container_width=True):
            st.session_state.pending_new_resume = [new_file]
            st.rerun()

    if st.button("🚪 Logout", use_container_width=True):
        for key in ["auth_user", "auth_profile", "vectorstore", "splits", "all_embeddings"]:
            st.session_state.pop(key, None)
        st.rerun()

if "pending_new_resume" in st.session_state:
    _confirm_reprocess_dialog()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Visual RAG Learning",
    "💬 Chat Assistant",
    "🔀 Hybrid Chat",
    "🪜 Level Answers",
    "📋 JD Answers"
])

with tab1:
    display_visual_pipeline()

with tab2:
    st.title("💬 StudySage Chat")
    st.write("Ask questions about your processed documents")
    display_chat_interface()

with tab3:
    display_hybrid_chat()

with tab4:
    display_level_chat()

with tab5:
    display_jd_prep()
