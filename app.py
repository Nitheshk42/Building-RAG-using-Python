import sys
try:
    # Cloud (Linux) environments often ship an sqlite3 too old for chromadb's rust
    # bindings - swap in pysqlite3-binary if it's installed. On macOS/local dev this
    # package usually isn't installed (or won't build), so just skip it silently -
    # local sqlite3 is normally recent enough already.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Compatibility shim: chromadb 0.4.24 still references numpy aliases that were removed
# in numpy 2.0 (np.float_, etc). If a numpy 2.x install slips through despite the
# requirements.txt pin (e.g. a cached build environment), restore them before any
# downstream library (chromadb) can hit the AttributeError.
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "complex_"):
    _np.complex_ = _np.complex128
if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

import streamlit as st
# NOTE: the tab modules (visual_pipeline, chat, hybrid_chat, level_chat, jd_chat) and
# onboarding are intentionally NOT imported here at module level. They pull in torch,
# sentence-transformers, chromadb, matplotlib, and scikit-learn - a genuinely heavy import
# chain. Importing them eagerly meant that chain had to fully load before this process could
# even bind to $PORT, which was causing connection timeouts on cold starts (Render's boot
# window, and similar issues elsewhere) - the health check couldn't get a response in time
# even though the actual app logic (login page) doesn't need any of that yet. They're
# imported lazily just below, right where each one is actually used, so the login/auth gate
# renders near-instantly and the heavy stack only loads once a user is authenticated and
# actually reaches a tab that needs it.
from src.auth_ui import display_auth
from src.auth import get_profile, save_feedback

st.set_page_config(page_title="StudySage", page_icon="📚", layout="wide")

# Google Sans Text (the actual font used on developers.google.com) + Roboto fallback,
# applied app-wide without breaking Streamlit's icon font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Text&family=Roboto:wght@300;400;500;700&display=swap');

html, body,
[data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
.stButton button, .stTextInput input, .stTextArea textarea, .stSelectbox,
.stRadio label, .stTabs, .stCaption, .stAlert,
h1, h2, h3, h4, h5, h6, p, li, label {
    font-family: 'Google Sans Text', 'Roboto', 'Segoe UI', sans-serif !important;
}

h1, h2, h3 {
    font-weight: 500 !important;
    letter-spacing: -0.3px;
}

/* Keep Streamlit's icon font working (upload icon, chevrons, etc.) - do not override these */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, span[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* ===== Visual polish: rounded cards, soft shadows, Google-blue accents ===== */

/* Bordered containers (answer cards across every tab) get a soft elevated look */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2) !important;
    transition: box-shadow 0.15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 12px rgba(26,115,232,0.15), 0 1px 3px rgba(0,0,0,0.3) !important;
}

/* Buttons: rounded, slightly bolder */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: transform 0.1s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
}

/* Tabs: cleaner underline, active tab in accent color */
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1a73e8 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #1a73e8 !important;
    height: 3px !important;
    border-radius: 2px !important;
}

/* Sidebar: subtle separation from main content */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Expanders: rounded, matching card style */
[data-testid="stExpander"] {
    border-radius: 10px !important;
    overflow: hidden;
}

/* Inputs: rounded corners for a softer feel */
.stTextInput input, .stTextArea textarea, .stSelectbox > div {
    border-radius: 8px !important;
}

/* Metrics: slight card feel */
[data-testid="stMetric"] {
    background: rgba(26,115,232,0.06);
    border-radius: 10px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

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
    from src.onboarding import display_onboarding
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
            from src.onboarding import process_resume
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

    from src.llm_provider import PROVIDERS, DEFAULT_PROVIDER
    provider_labels = list(PROVIDERS.keys())
    current_label = next((k for k, v in PROVIDERS.items() if v == st.session_state.get("llm_provider", DEFAULT_PROVIDER)), provider_labels[0])
    chosen_label = st.selectbox("🧠 Answer engine", provider_labels, index=provider_labels.index(current_label))
    st.session_state.llm_provider = PROVIDERS[chosen_label]
    st.caption("Switches which LLM generates answers across every tab.")

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

    st.divider()
    with st.expander("💬 Send feedback"):
        feedback_text = st.text_area(
            "What's working, what's not, what would help?",
            key="feedback_text", height=100, label_visibility="collapsed",
            placeholder="What's working, what's not, what would help?"
        )
        if st.button("Submit feedback", use_container_width=True):
            if save_feedback(st.session_state.auth_user, feedback_text):
                st.success("✅ Thanks — feedback saved!")
            else:
                st.warning("Write something before submitting.")

if "pending_new_resume" in st.session_state:
    _confirm_reprocess_dialog()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📖 Visual RAG Learning",
    "💬 Chat Assistant",
    "🔀 Hybrid Chat",
    "🪜 My EXP Level Answers",
    "📋 My JD Answers",
    "🧠 General JD Answers",
    "🎯 Resume Tailor"
])

with tab1:
    from src.visual_pipeline import display_visual_pipeline
    display_visual_pipeline()

with tab2:
    from src.chat import display_chat_interface
    st.title("💬 StudySage Chat")
    st.write("Ask questions about your processed documents")
    display_chat_interface()

with tab3:
    from src.hybrid_chat import display_hybrid_chat
    display_hybrid_chat()

with tab4:
    from src.level_chat import display_level_chat
    display_level_chat()

with tab5:
    from src.jd_chat import display_jd_prep
    display_jd_prep()

with tab6:
    from src.general_jd_chat import display_general_jd_prep
    display_general_jd_prep()

with tab7:
    from src.resume_tailor_ui import display_resume_tailor
    display_resume_tailor()
