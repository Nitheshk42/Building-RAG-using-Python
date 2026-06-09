import streamlit as st
from src.visual_pipeline import display_visual_pipeline
from src.chat import display_chat_interface

st.set_page_config(page_title="StudySage", page_icon="📚", layout="wide")

# Simple tab navigation
tab1, tab2 = st.tabs(["📖 Visual RAG Learning (Day 1)", "💬 Chat Assistant (Day 2)"])

with tab1:
    display_visual_pipeline()

with tab2:
    st.title("💬 StudySage Chat")
    st.write("Ask questions about your processed documents")
    display_chat_interface()