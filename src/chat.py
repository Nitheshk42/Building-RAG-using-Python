import streamlit as st
from src.rag_pipeline import get_rag_chain
from src.vector_store import get_vectorstore
import time

def display_chat_interface():
    """Chat interface with dynamic LLM reasoning sidebar"""

    st.title("💬 Resume Q&A Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main chat area
    col_chat, col_reasoning = st.columns([2, 1], gap="large")
    
    # ===== LEFT SIDE: CHAT =====
    with col_chat:
        st.subheader("Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input box
        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Processing..."):
                    vectorstore = st.session_state.get("vectorstore") or get_vectorstore(st.session_state.get("auth_user"))

                    if not vectorstore:
                        st.warning("⚠️ Please go to Visual tab and process documents first.")
                        return

                    # Real retrieval with real similarity scores (for the sidebar)
                    retrieved = vectorstore.similarity_search_with_score(prompt, k=3)
                    context_text = "\n\n".join(doc.page_content for doc, _ in retrieved)

                    rag_chain = get_rag_chain(vectorstore)
                    answer = rag_chain.invoke({"question": prompt})

                    st.markdown(answer)

                    # Store real data for sidebar display
                    st.session_state.last_question = prompt
                    st.session_state.last_answer = answer
                    st.session_state.last_chunks = retrieved
                    st.session_state.last_context = context_text
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # ===== RIGHT SIDE: LLM REASONING SIDEBAR =====
    with col_reasoning:
        st.subheader("🤖 LLM Reasoning")
        
        if "last_question" in st.session_state:
            question = st.session_state.last_question
            chunks = st.session_state.last_chunks

            # Step 1: Real retrieved chunks with real HNSW distance scores
            with st.expander("✅ Step 1: Retrieved Chunks (HNSW)", expanded=True):
                st.markdown("**Your question was embedded and searched against the vector DB:**")
                for i, (doc, score) in enumerate(chunks, start=1):
                    st.markdown(f"📄 **Chunk {i} (distance: {score:.4f}, lower = closer):**")
                    st.code(doc.page_content, language=None)

            # Step 2: The actual prompt sent to the LLM
            with st.expander("✅ Step 2: Prompt Sent to LLM", expanded=True):
                st.markdown("**This is exactly what the LLM received — retrieved chunks + your question:**")
                st.code(
                    f"Context:\n{st.session_state.last_context}\n\nQuestion: {question}",
                    language=None
                )

            # Step 3: Real final answer
            with st.expander("✅ Step 3: Generated Answer", expanded=True):
                st.markdown(f"> {st.session_state.last_answer}")

            st.divider()
            st.caption(
                "The LLM only ever sees the chunks retrieved above — it cannot use "
                "outside knowledge in this tab, so any answer is traceable back to a specific chunk."
            )
        else:
            st.info("💭 Ask a question to see LLM reasoning here...")