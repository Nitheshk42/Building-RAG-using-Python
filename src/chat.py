import streamlit as st
from src.rag_pipeline import get_rag_chain
from src.vector_store import get_vectorstore

def display_chat_interface():
    """Clean chat interface for Day 2"""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                vectorstore = st.session_state.get("vectorstore") or get_vectorstore()
                
                if not vectorstore:
                    st.warning("⚠️ Please go to Visual tab and process documents first.")
                    return
                
                rag_chain = get_rag_chain(vectorstore)
                answer = rag_chain.invoke({"question": prompt})  # Returns string directly
                
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})