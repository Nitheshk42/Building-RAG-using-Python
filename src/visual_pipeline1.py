import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store

def display_visual_pipeline1():
    """All visual learning stages - Day 1"""
    st.title("📚 Visual RAG Pipeline - How It Actually Works")
    st.write("Watch step-by-step how your documents become searchable")

    # Sidebar Upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            if st.button("💾 Save Files to Data Folder"):
                os.makedirs("data", exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("✅ All files saved to `data/` folder!")

    # Stage Selection
    stage = st.radio("Go to stage:", [
        "1️⃣ Chunking", "2️⃣ Embeddings", "3️⃣ Vector DB", "4️⃣ HNSW Search"
    ], horizontal=True)

    # ======================== STAGE 1: CHUNKING ========================
    if stage == "1️⃣ Chunking":
        st.header("Stage 1: Document Chunking")
        st.write("Split documents into overlapping chunks (1000 chars each)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("📖 See full document"):
                if os.path.exists("data") and os.listdir("data"):
                    docs = load_documents()
                    if docs:
                        st.write(docs[0].page_content[:1000])
        
        with col2:
            if st.button("✂️ Chunk Document", use_container_width=True):
                if not os.path.exists("data") or len(os.listdir("data")) == 0:
                    st.error("❌ No PDFs in data/ folder")
                else:
                    documents = load_documents()
                    splits = split_documents(documents)
                    st.session_state.splits = splits
                    st.success(f"✅ Split into {len(splits)} chunks")
        
        if 'splits' in st.session_state:
            splits = st.session_state.splits
            fig, ax = plt.subplots(figsize=(20, 4))
            chunk_lengths = [len(s.page_content) for s in splits[:20]]
            colors = ['#667eea' if i % 2 == 0 else '#764ba2' for i in range(len(chunk_lengths))]
            ax.barh(range(len(chunk_lengths)), chunk_lengths, color=colors, edgecolor='black')
            ax.set_xlabel("Characters per Chunk")
            ax.axvline(1000, color='red', linestyle='--', label='Target')
            ax.legend()
            st.pyplot(fig)

    # ======================== STAGE 2: EMBEDDINGS ========================
    elif stage == "2️⃣ Embeddings":
        st.header("Stage 2: Convert to Embeddings")
        if st.button("🧠 Create Embeddings", use_container_width=True):
            if 'splits' not in st.session_state:
                st.error("Complete Stage 1 first!")
            else:
                splits = st.session_state.splits
                embeddings_model = get_embeddings()
                all_embeddings = [embeddings_model.embed_query(s.page_content[:300]) for s in splits]
                all_embeddings = np.array(all_embeddings)
                st.session_state.all_embeddings = all_embeddings
                st.session_state.embeddings_model = embeddings_model
                st.success(f"✅ Created {len(all_embeddings)} vectors!")

        if 'all_embeddings' in st.session_state:
            all_emb = st.session_state.all_embeddings
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb)-1))
            embeddings_2d = tsne.fit_transform(all_emb)
            fig, ax = plt.subplots(figsize=(25, 8))
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(embeddings_2d)), cmap='viridis', s=150)
            ax.set_title("Embedding Space (t-SNE)")
            st.pyplot(fig)

    # ======================== STAGE 3: VECTOR DB ========================
    elif stage == "3️⃣ Vector DB":
        st.header("Stage 3: Save to Vector Database")
        if st.button("💾 Create Vector Database", use_container_width=True):
            if 'splits' not in st.session_state:
                st.error("Complete Stage 1 first!")
            else:
                splits = st.session_state.splits
                embeddings_model = get_embeddings()
                vectorstore = create_vector_store(splits, embeddings_model)
                st.session_state.vectorstore = vectorstore
                st.success("✅ Vector DB created!")

    # ======================== STAGE 4: HNSW SEARCH ========================
    elif stage == "4️⃣ HNSW Search":
        st.header("Stage 4: Search with HNSW")
        if 'vectorstore' not in st.session_state:
            st.error("Complete Stage 3 first!")
        else:
            user_query = st.text_input("Ask a question:", "What is my tech stack?")
            if st.button("🔍 Search", use_container_width=True):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)
                
                st.subheader("📌 Top 3 Most Relevant Chunks")
                for i, doc in enumerate(retrieved_docs, 1):
                    with st.expander(f"✅ Result {i}", expanded=True):
                        st.write(doc.page_content)

    st.divider()
    st.info("**Visual Learning Summary:** Complete all stages to understand the RAG pipeline.")