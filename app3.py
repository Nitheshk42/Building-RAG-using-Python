import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store

st.set_page_config(page_title="Visual RAG Learning", layout="wide")

st.title("📚 Visual RAG Pipeline - How It Actually Works")
st.write("Watch step-by-step how your documents become searchable")

# Sidebar for navigation
with st.sidebar:
    st.header("📍 Navigation")
    stage = st.radio("Go to stage:", [
        "1️⃣ Chunking",
        "2️⃣ Embeddings", 
        "3️⃣ Vector DB",
        "4️⃣ HNSW Search"
    ])
    st.header("📤 Upload Documents")
    st.write("Upload your lecture notes / PDFs")
    
    uploaded_files = st.file_uploader(
       "Choose PDF files", 
        type="pdf", 
         accept_multiple_files=True
    )
    
    if uploaded_files:
         st.success(f"✅ {len(uploaded_files)} file(s) selected")
         if st.button("💾 Save Files to Data Folder"):
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"✅ All files saved to `data/` folder!")

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
                st.session_state.documents = documents
                
                st.success(f"✅ Split into {len(splits)} chunks")
                st.metric("Chunks Created", len(splits))
    
    if 'splits' in st.session_state:
        st.subheader("Chunk Visualization")
        splits = st.session_state.splits
        
        fig, ax = plt.subplots(figsize=(20, 4))
        chunk_lengths = [len(s.page_content) for s in splits[:20]]
        colors = ['#667eea' if i % 2 == 0 else '#764ba2' for i in range(len(chunk_lengths))]
        
        ax.barh(range(len(chunk_lengths)), chunk_lengths, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_xlabel("Characters per Chunk")
        ax.set_ylabel("Chunk Number")
        ax.set_title("Chunk Sizes (overlapping pieces)", fontweight='bold')
        ax.axvline(1000, color='red', linestyle='--', linewidth=2, label='Target: 1000 chars')
        ax.legend()
        
        st.pyplot(fig)
        
        st.write("**First 3 chunks:**")
        for i in range(min(3, len(splits))):
            with st.expander(f"Chunk {i+1} ({len(splits[i].page_content)} chars)"):
                st.text(splits[i].page_content[:300] + "...")

# ======================== STAGE 2: EMBEDDINGS ========================
elif stage == "2️⃣ Embeddings":
    st.header("Stage 2: Convert to Embeddings")
    st.write("Transform each chunk into a 384-dimensional vector (captures meaning)")
    
    if st.button("🧠 Create Embeddings", use_container_width=True):
        if 'splits' not in st.session_state:
            st.error("❌ Complete Stage 1 first!")
        else:
            splits = st.session_state.splits
            embeddings_model = get_embeddings()
            
            # Create embeddings
            all_embeddings = []
            progress_bar = st.progress(0)
            for idx, split in enumerate(splits):
                emb = embeddings_model.embed_query(split.page_content[:300])
                all_embeddings.append(emb)
                progress_bar.progress((idx + 1) / len(splits))
            
            all_embeddings = np.array(all_embeddings)
            st.session_state.embeddings_model = embeddings_model
            st.session_state.all_embeddings = all_embeddings
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🧠 Embedding Model", "Sentence\nTransformer")
            col2.metric("📐 Vector\nDimension", "384")
            col3.metric("📦 Vectors\nCreated", len(all_embeddings))
            
            st.success(f"✅ Created {len(all_embeddings)} vectors!")
    
    if 'all_embeddings' in st.session_state:
        st.subheader("Embedding Visualization (t-SNE 2D)")
        st.write("Each point = one chunk as a vector (reduced to 2D for visualization)")
        
        all_emb = st.session_state.all_embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb)-1))
        embeddings_2d = tsne.fit_transform(all_emb)
        
        fig, ax = plt.subplots(figsize=(25, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=range(len(embeddings_2d)), cmap='viridis', 
                            s=150, alpha=0.7, edgecolors='black', linewidth=1)
        ax.set_title("Vector Space: Similar Chunks Close Together", fontweight='bold', fontsize=12)
        ax.set_xlabel("Semantic Dimension 1")
        ax.set_ylabel("Semantic Dimension 2")
        plt.colorbar(scatter, ax=ax, label='Chunk Number')
        
        st.pyplot(fig)
        
        st.write("**Sample embedding (first 10 dimensions):**")
        st.code(f"[{', '.join([f'{x:.3f}' for x in all_emb[0][:10]])}...]")

# ======================== STAGE 3: VECTOR DB ========================
elif stage == "3️⃣ Vector DB":
    st.header("Stage 3: Save to Vector Database with HNSW Index")
    st.write("Store vectors using HNSW (fast multi-layer search structure)")
    
    if st.button("💾 Create Vector Database", use_container_width=True):
        if 'splits' not in st.session_state:
            st.error("❌ Complete Stage 1 first!")
        else:
            splits = st.session_state.splits
            embeddings_model = get_embeddings()
            
            vectorstore = create_vector_store(splits, embeddings_model)
            st.session_state.vectorstore = vectorstore
            st.session_state.embeddings_model = embeddings_model
            
            col1, col2 = st.columns(2)
            col1.metric("💾 Database Type", "ChromaDB")
            col2.metric("⚡ Index Type", "HNSW")
            
            st.success("✅ Vector DB created!")
    
    if 'vectorstore' in st.session_state:
        st.subheader("HNSW Layer Structure")
        st.write("Multi-layer index: use highways for speed, small roads for precision")
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        np.random.seed(42)
        
        # Top Layer (sparse)
        ax = axes[0]
        points_top = np.random.rand(5, 2) * 10
        ax.scatter(points_top[:, 0], points_top[:, 1], s=300, c='gold', edgecolors='orange', linewidth=2)
        for i in range(len(points_top)):
            for j in range(i+1, len(points_top)):
                ax.plot([points_top[i,0], points_top[j,0]], [points_top[i,1], points_top[j,1]], 
                       'orange', linewidth=2, alpha=0.6)
        ax.set_title("Top Layer\n(Entry Points)\nFast Navigation", fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
        
        # Middle Layer
        ax = axes[1]
        points_mid = np.random.rand(12, 2) * 10
        ax.scatter(points_mid[:, 0], points_mid[:, 1], s=200, c='lightblue', edgecolors='blue', linewidth=1.5)
        for i in range(min(6, len(points_mid))):
            for j in range(i+1, min(6, len(points_mid))):
                if np.linalg.norm(points_mid[i] - points_mid[j]) < 4:
                    ax.plot([points_mid[i,0], points_mid[j,0]], [points_mid[i,1], points_mid[j,1]], 
                           'blue', linewidth=0.8, alpha=0.4)
        ax.set_title("Middle Layer\n(Navigation)\nBalanced", fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
        
        # Bottom Layer (all vectors)
        ax = axes[2]
        points_bot = np.random.rand(30, 2) * 10
        ax.scatter(points_bot[:, 0], points_bot[:, 1], s=100, c='lightcoral', edgecolors='red', linewidth=1)
        for i in range(min(10, len(points_bot))):
            for j in range(i+1, min(10, len(points_bot))):
                if np.linalg.norm(points_bot[i] - points_bot[j]) < 2:
                    ax.plot([points_bot[i,0], points_bot[j,0]], [points_bot[i,1], points_bot[j,1]], 
                           'red', linewidth=0.3, alpha=0.2)
        ax.set_title("Bottom Layer\n(All Vectors)\nComplete Search", fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        🟠 **Top Layer**: Few entry points, connected by "highways" → Fast coarse search
        🔵 **Middle Layer**: More points, selective connections → Refinement  
        🔴 **Bottom Layer**: All vectors, local connections → Final precise search
        """)

# ======================== STAGE 4: HNSW SEARCH ========================
elif stage == "4️⃣ HNSW Search":
    st.header("Stage 4: Search with HNSW - See How It Finds Answers")
    st.write("Upload resume → Ask question → Watch HNSW find relevant chunks")
    
    if 'vectorstore' not in st.session_state:
        st.error("❌ Complete all previous stages first!")
    else:
        # Get user query
        user_query = st.text_input(
            "Ask a question about your document:",
            "What is my tech stack?",
            placeholder="e.g., What skills do I have?"
        )
        
        if st.button("🔍 Search & Visualize", use_container_width=True):
            splits = st.session_state.splits
            embeddings_model = st.session_state.embeddings_model
            all_embeddings = st.session_state.all_embeddings
            vectorstore = st.session_state.vectorstore
            
            # Get query embedding
            query_embedding = embeddings_model.embed_query(user_query)
            
            # Retrieve results
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(user_query)
            
            # ===== VISUALIZATION 1: HNSW Search Process =====
            st.subheader("🎯 How HNSW Found Your Answer")
            
            # Reduce to 2D
            sample_size = min(30, len(all_embeddings))
            sample_embeddings = all_embeddings[:sample_size]
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
            embeddings_2d = tsne.fit_transform(sample_embeddings)
            
            fig, ax = plt.subplots(figsize=(25, 8))
            
            # All chunks (blue)
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightblue', s=200, 
                      label='All Chunks', edgecolors='blue', linewidth=2, alpha=0.6, zorder=3)
            
            # Your query (red star)
            query_2d = np.array([embeddings_2d.mean(axis=0)])  # Query position
            ax.scatter(query_2d[0, 0], query_2d[0, 1], c='red', s=600, marker='*', 
                      label='Your Question', edgecolors='darkred', linewidth=2, zorder=5)
            
            # HNSW Highways (orange - long connections)
            st.write("**Highways (orange lines)**: Fast long-distance navigation")
            for i in range(min(3, len(embeddings_2d))):
                for j in range(i+1, min(3, len(embeddings_2d))):
                    ax.plot([embeddings_2d[i,0], embeddings_2d[j,0]], 
                           [embeddings_2d[i,1], embeddings_2d[j,1]], 
                           'orange', linewidth=3, alpha=0.7, label='Highways' if i==0 and j==1 else '')
            
            # Small roads (gray - local connections)
            st.write("**Small roads (gray lines)**: Local neighbor connections")
            for i in range(len(embeddings_2d)):
                distances = np.linalg.norm(embeddings_2d - embeddings_2d[i], axis=1)
                nearest_idx = np.argsort(distances)[1:4]
                for j in nearest_idx:
                    ax.plot([embeddings_2d[i,0], embeddings_2d[j,0]], 
                           [embeddings_2d[i,1], embeddings_2d[j,1]], 
                           'gray', linewidth=0.8, alpha=0.3)
            
            # Found chunks (green dots with paths)
            st.write("**Found chunks (green)**: Most relevant to your question")
            for idx in range(min(3, len(embeddings_2d))):
                ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c='lime', s=350, 
                          marker='o', edgecolors='darkgreen', linewidth=2, zorder=6)
                # Draw path from query to found chunk
                ax.annotate('', xy=(embeddings_2d[idx, 0], embeddings_2d[idx, 1]), 
                           xytext=(query_2d[0, 0], query_2d[0, 1]),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7))
            
            ax.set_title("HNSW Search: Finding Relevant Chunks via Highways & Roads", 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel("Vector Space Dimension 1")
            ax.set_ylabel("Vector Space Dimension 2")
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
            
            # ===== VISUALIZATION 2: Search Process Steps =====
            st.subheader("🚀 Search Process Steps")
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 3))
            
            # Step 1: Start at top
            ax = axes[0]
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightblue', s=150, alpha=0.4)
            ax.scatter([embeddings_2d[0, 0]], [embeddings_2d[0, 1]], c='gold', s=300, 
                      edgecolors='orange', linewidth=2, label='Entry Point')
            ax.set_title("Step 1: Enter at Top Layer\n(Start here)", fontweight='bold')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            
            # Step 2: Navigate top layer
            ax = axes[1]
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightblue', s=150, alpha=0.4)
            for i in range(3):
                ax.plot([embeddings_2d[i,0], embeddings_2d[(i+1)%3,0]], 
                       [embeddings_2d[i,1], embeddings_2d[(i+1)%3,1]], 
                       'orange', linewidth=2)
            ax.scatter(embeddings_2d[:3, 0], embeddings_2d[:3, 1], c='gold', s=250, edgecolors='orange')
            ax.set_title("Step 2: Use Highways\n(Fast navigation)", fontweight='bold')
            ax.grid(alpha=0.2)
            
            # Step 3: Descend to middle/bottom
            ax = axes[2]
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightblue', s=150, alpha=0.5)
            for i in range(len(embeddings_2d)):
                distances = np.linalg.norm(embeddings_2d - embeddings_2d[i], axis=1)
                nearest = np.argsort(distances)[1]
                ax.plot([embeddings_2d[i,0], embeddings_2d[nearest,0]], 
                       [embeddings_2d[i,1], embeddings_2d[nearest,1]], 
                       'gray', linewidth=0.5, alpha=0.3)
            ax.set_title("Step 3: Use Small Roads\n(Refine locally)", fontweight='bold')
            ax.grid(alpha=0.2)
            
            # Step 4: Found!
            ax = axes[3]
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightblue', s=150, alpha=0.4)
            ax.scatter(embeddings_2d[:3, 0], embeddings_2d[:3, 1], c='lime', s=300, 
                      edgecolors='darkgreen', linewidth=2, label='Top 3 Results')
            ax.set_title("Step 4: Found Chunks!\n(Most relevant)", fontweight='bold')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ===== RESULTS =====
            st.subheader("📌 Top 3 Most Relevant Chunks")
            
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"✅ Result {i}", expanded=(i==1)):
                    st.write(doc.page_content)

st.divider()
st.info("""
**Visual Learning Summary:**
1. **Stage 1 - Chunking**: Break document into overlapping pieces
2. **Stage 2 - Embeddings**: Each chunk → 384D vector (semantic meaning)
3. **Stage 3 - Vector DB**: Store vectors with HNSW multi-layer structure
4. **Stage 4 - Search**: Question → Use highways (fast) + roads (precise) → Find answer
""")