import streamlit as st
from src.visual_pipeline import display_visual_pipeline
from src.chat import display_chat_interface
from src.hybrid_chat import display_hybrid_chat
from src.level_chat import display_level_chat
from src.jd_chat import display_jd_prep

st.set_page_config(page_title="StudySage", page_icon="📚", layout="wide")

# Tab navigation
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
