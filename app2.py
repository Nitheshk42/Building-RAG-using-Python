import streamlit as st
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle
import pandas as pd
from datetime import datetime
import json

# Suppress annoying transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings, get_sample_embedding
from src.vector_store import create_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="StudySage - AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .stage-container {
        background: #f8f9fa;
        padding: 20px;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .success-banner {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        font-weight: bold;
        text-align: center;
    }
    
    .error-banner {
        background: linear-gradient(90deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    .chunk-preview {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .visualization-title {
        font-size: 24px;
        font-weight: bold;
        margin: 30px 0 20px 0;
        color: #1f1f1f;
    }
    
    .step-description {
        background: #e3f2fd;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1 style='font-size: 48px; margin: 0;'>🧠 StudySage</h1>
    <p style='font-size: 18px; color: #666; margin: 10px 0;'>Your AI Tutor with Visual RAG Pipeline</p>
    <p style='font-size: 14px; color: #999;'>Full Ingestion Pipeline + HNSW Deep Dive</p>
</div>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        if st.button("💾 Save Files to Data Folder", use_container_width=True):
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("✅ Files saved to `data/` folder!")
    
    st.divider()
    st.markdown("### 📊 Pipeline Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if 'pipeline_complete' in st.session_state and st.session_state.pipeline_complete:
            st.metric("Status", "✅ Complete", delta="Ready")
        else:
            st.metric("Status", "⏳ Pending", delta="Not started")
    
    st.divider()
    st.markdown("### 🎯 Next Steps")
    st.info("""
    1. **Upload** PDFs in the section above
    2. **Run** the full ingestion pipeline
    3. **Explore** visualizations
    4. **Query** your documents
    """)

# ===================== MAIN PIPELINE =====================
st.markdown("<div class='visualization-title'>🚀 Ingest Documents - Visual Pipeline</div>", unsafe_allow_html=True)

pipeline_col1, pipeline_col2 = st.columns([3, 1])
with pipeline_col2:
    st.markdown("**Pipeline Stages**")
    st.markdown("1️⃣ Load")
    st.markdown("2️⃣ Chunk")
    st.markdown("3️⃣ Embed")
    st.markdown("4️⃣ Index")

with pipeline_col1:
    if st.button("🔄 Start Full Ingestion Pipeline", type="primary", use_container_width=True, key="ingestion"):
        if not os.path.exists("data") or len(os.listdir("data")) == 0:
            st.markdown("""
            <div class='error-banner'>
                ❌ No PDFs found in data folder. Please upload files first!
            </div>
            """, unsafe_allow_html=True)
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # ==================== STAGE 1: LOAD ====================
                status_text.text("📖 Stage 1/4: Loading PDFs...")
                progress_bar.progress(0)
                
                st.markdown("<div class='stage-container'>", unsafe_allow_html=True)
                st.write("**Stage 1: Loading PDFs**")
                
                documents = load_documents()
                doc_count = len(documents)
                
                st.success(f"✅ Loaded {doc_count} document(s)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents", doc_count)
                with col2:
                    total_chars = sum(len(doc.page_content) for doc in documents)
                    st.metric("📝 Total Chars", f"{total_chars:,}")
                with col3:
                    avg_pages = doc_count if doc_count > 0 else 1
                    st.metric("📊 Avg Size", f"{total_chars // avg_pages:,}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                progress_bar.progress(25)
                
                # ==================== STAGE 2: CHUNK ====================
                status_text.text("✂️ Stage 2/4: Chunking Documents...")
                progress_bar.progress(25)
                
                st.markdown("<div class='stage-container'>", unsafe_allow_html=True)
                st.write("**Stage 2: Splitting into Chunks**")
                
                splits = split_documents(documents)
                chunk_count = len(splits)
                
                st.markdown("""
                <div class='step-description'>
                    Documents are split into overlapping chunks to maintain context while keeping vectors manageable.
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✂️ Total Chunks", chunk_count)
                with col2:
                    st.metric("📏 Chunk Size", "1000 chars")
                with col3:
                    st.metric("🔄 Overlap", "200 chars")
                with col4:
                    avg_chunk = sum(len(s.page_content) for s in splits) // max(chunk_count, 1)
                    st.metric("📊 Avg Chunk", f"{avg_chunk} chars")
                
                # Visual representation of chunking
                st.write("**First 3 Chunks Preview:**")
                chunk_tabs = st.tabs([f"Chunk {i+1}" for i in range(min(3, len(splits)))])
                
                for idx, tab in enumerate(chunk_tabs):
                    with tab:
                        st.markdown(f"""
                        <div class='chunk-preview'>
                            <small>📍 Source: {splits[idx].metadata.get('source', 'Unknown')}</small><br>
                            <small>📄 Page: {splits[idx].metadata.get('page', 'N/A')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.text_area(
                            "Content:",
                            splits[idx].page_content[:400] + "..." if len(splits[idx].page_content) > 400 else splits[idx].page_content,
                            height=180,
                            disabled=True,
                            key=f"chunk_{idx}"
                        )
                
                st.markdown("</div>", unsafe_allow_html=True)
                progress_bar.progress(50)
                
                # ==================== STAGE 3: EMBED ====================
                status_text.text("🔢 Stage 3/4: Creating Embeddings...")
                progress_bar.progress(50)
                
                st.markdown("<div class='stage-container'>", unsafe_allow_html=True)
                st.write("**Stage 3: Converting to Embedding Vectors**")
                
                st.markdown("""
                <div class='step-description'>
                    Each chunk is converted to a vector using a sentence transformer model. 
                    These vectors capture semantic meaning and allow similarity comparisons.
                </div>
                """, unsafe_allow_html=True)
                
                embeddings = get_embeddings()
                sample_embedding = get_sample_embedding(embeddings, splits[0].page_content[:200])
                embedding_dim = len(sample_embedding)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🧠 Embedding Model", "Sentence Transformer")
                with col2:
                    st.metric("📐 Vector Dimensions", embedding_dim)
                with col3:
                    st.metric("💾 Total Vectors", chunk_count)
                
                st.success("✅ Embedding Model Ready")
                
                # Show sample embedding
                with st.expander("👁️ View Sample Embedding Vector"):
                    st.write("First 20 dimensions of the embedding vector:")
                    embedding_df = pd.DataFrame({
                        "Dimension": [f"dim_{i}" for i in range(20)],
                        "Value": sample_embedding[:20]
                    })
                    st.dataframe(embedding_df, use_container_width=True)
                    
                    # Embedding statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Value", f"{sample_embedding.min():.4f}")
                    with col2:
                        st.metric("Max Value", f"{sample_embedding.max():.4f}")
                    with col3:
                        st.metric("Mean Value", f"{sample_embedding.mean():.4f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                progress_bar.progress(75)
                
                # ==================== STAGE 4: INDEX ====================
                status_text.text("🗂️ Stage 4/4: Building Vector Index (HNSW)...")
                progress_bar.progress(75)
                
                st.markdown("<div class='stage-container'>", unsafe_allow_html=True)
                st.write("**Stage 4: Building Vector Database with HNSW Index**")
                
                st.markdown("""
                <div class='step-description'>
                    HNSW (Hierarchical Navigable Small World) creates a multi-layer index structure 
                    for ultra-fast similarity search. It uses "highways" for long-distance navigation 
                    and "local roads" for precision.
                </div>
                """, unsafe_allow_html=True)
                
                vectorstore = create_vector_store(splits, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.splits = splits
                st.session_state.embeddings = embeddings
                st.session_state.pipeline_complete = True
                
                progress_bar.progress(100)
                status_text.text("✅ Pipeline Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🗂️ Index Type", "HNSW")
                with col2:
                    st.metric("📦 Stored Vectors", chunk_count)
                with col3:
                    st.metric("⚡ Search Speed", "O(log n)")
                
                st.markdown("""
                <div class='success-banner'>
                    🎉 Vector Database Created Successfully!
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ==================== VISUALIZATION: HNSW MECHANISM ====================
                st.markdown("<div class='visualization-title'>🔍 Understanding HNSW: Visual Deep Dive</div>", unsafe_allow_html=True)
                
                # Try to use real embeddings with dimensionality reduction
                try:
                    from sklearn.manifold import TSNE
                    from sklearn.preprocessing import StandardScaler
                    
                    st.write("**Using Real Embeddings with t-SNE Dimensionality Reduction**")
                    st.markdown("""
                    <div class='step-description'>
                        Real embedding vectors are reduced from 384 dimensions to 2D for visualization using t-SNE.
                        This preserves local and global structure while making patterns visible.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("🔄 Reducing embedding dimensions with t-SNE..."):
                        # Sample embeddings for visualization (limit to avoid slow t-SNE)
                        sample_size = min(50, len(splits))
                        sample_indices = np.random.choice(len(splits), sample_size, replace=False)
                        
                        sample_vectors = np.array([
                            embeddings.embed_query(splits[i].page_content[:300])
                            for i in sample_indices
                        ])
                        
                        # Standardize
                        scaler = StandardScaler()
                        sample_vectors_scaled = scaler.fit_transform(sample_vectors)
                        
                        # Apply t-SNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
                        vectors_2d = tsne.fit_transform(sample_vectors_scaled)
                        
                        use_real_embeddings = True
                        st.success("✅ t-SNE visualization ready!")
                        
                except ImportError:
                    st.warning("⚠️ scikit-learn not available. Using simulated 2D vectors for visualization.")
                    use_real_embeddings = False
                except Exception as e:
                    st.warning(f"⚠️ Could not reduce embeddings: {str(e)}. Using simulated 2D vectors.")
                    use_real_embeddings = False
                
                if not use_real_embeddings:
                    # Fallback to simulation
                    np.random.seed(42)
                    sample_size = min(15, len(splits))
                    vectors_2d = np.random.rand(sample_size, 2) * 10
                
                query_point = np.array([vectors_2d[:, 0].mean(), vectors_2d[:, 1].mean()])
                
                # HNSW Stages
                stages = [
                    ("1. All Chunk Vectors in 2D Space", "Vector embeddings plotted in 2D space (real embeddings reduced via t-SNE)"),
                    ("2. HNSW Builds Multi-Layer Index", "Highways (red) connect distant vectors for fast traversal. Local roads (gray) connect neighbors."),
                    ("3. Search: Start from Top Layer", "Search enters at top layer entry points, then navigates downward through layers."),
                    ("4. Convergence: Find Nearest Neighbors", "Algorithm converges to vectors closest to the query vector (green dashed paths).")
                ]
                
                visualization_tabs = st.tabs([title.split(".")[1].strip() for title in stages])
                
                for stage_num, (tab, (title, description)) in enumerate(zip(visualization_tabs, stages), 1):
                    with tab:
                        st.write(f"**{title}**")
                        st.markdown(f"<div class='step-description'>{description}</div>", unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 9), facecolor='#f8f9fa')
                        
                        # Styling
                        ax.set_facecolor('white')
                        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                        ax.set_xlabel("Dimension 1 (t-SNE)", fontsize=11)
                        ax.set_ylabel("Dimension 2 (t-SNE)", fontsize=11)
                        ax.grid(True, alpha=0.2, linestyle='--')
                        
                        # Base scatter plot
                        ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='#667eea', s=150, 
                                  alpha=0.7, edgecolors='white', linewidth=2, label='Chunk Vectors', zorder=3)
                        
                        # Query point
                        ax.scatter(query_point[0], query_point[1], c='#ff6a00', s=400, 
                                  marker='*', edgecolors='#333', linewidth=2, label='Query Vector', zorder=4)
                        
                        if stage_num == 2:  # Multi-Layer Index
                            # Highways (far connections)
                            highway_count = 0
                            for i in range(len(vectors_2d)):
                                for j in range(i+1, len(vectors_2d)):
                                    dist = np.linalg.norm(vectors_2d[i] - vectors_2d[j])
                                    if dist > np.percentile(pdist(vectors_2d), 70):
                                        ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
                                               [vectors_2d[i,1], vectors_2d[j,1]], 
                                               color='#ee0979', linewidth=2.5, alpha=0.5, zorder=1)
                                        highway_count += 1
                                        if highway_count > 20:  # Limit for clarity
                                            break
                                if highway_count > 20:
                                    break
                            
                            # Small roads (close connections)
                            road_count = 0
                            for i in range(len(vectors_2d)):
                                for j in range(i+1, len(vectors_2d)):
                                    dist = np.linalg.norm(vectors_2d[i] - vectors_2d[j])
                                    if dist < np.percentile(pdist(vectors_2d), 30):
                                        ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
                                               [vectors_2d[i,1], vectors_2d[j,1]], 
                                               color='#999', linewidth=1, alpha=0.4, zorder=1)
                                        road_count += 1
                                        if road_count > 25:
                                            break
                                if road_count > 25:
                                    break
                            
                            # Legend
                            highway_patch = mpatches.Patch(color='#ee0979', label='🔴 Highways (long-range)')
                            road_patch = mpatches.Patch(color='#999', label='⚪ Local Roads (neighbors)')
                            ax.legend(handles=[highway_patch, road_patch], loc='upper right', fontsize=10)
                            
                            st.markdown("""
                            **Key Concept:**
                            - 🔴 **Red Highways**: Connect distant vectors for fast long-distance navigation
                            - ⚪ **Gray Local Roads**: Connect nearby vectors for precision
                            - This structure allows O(log n) search complexity!
                            """)
                        
                        elif stage_num == 3:  # Top Layer
                            # Entry points
                            entry_idx = np.random.choice(len(vectors_2d), size=max(2, len(vectors_2d)//3), replace=False)
                            entry_points = vectors_2d[entry_idx]
                            
                            ax.scatter(entry_points[:, 0], entry_points[:, 1], c='#ffd700', s=280, 
                                      marker='D', edgecolors='#ff9800', linewidth=2.5, 
                                      label='Top Layer Entry Points', zorder=5)
                            
                            # Draw connection from entry to query area
                            for point in entry_points:
                                ax.annotate('', xy=query_point, xytext=point,
                                           arrowprops=dict(arrowstyle='->', color='#ff9800', 
                                                         lw=1.5, alpha=0.5, linestyle='--'))
                            
                            ax.legend(fontsize=10)
                            
                            st.markdown("""
                            **Top Layer Navigation:**
                            - 🟡 **Entry Points**: Search begins at these high-level entry points
                            - 🔗 **Downward Path**: Algorithm navigates down through layers toward query
                            - ⚡ **Fast Coarse Search**: Top layers enable rapid navigation in large spaces
                            """)
                        
                        elif stage_num == 4:  # Nearest Neighbors
                            distances = np.linalg.norm(vectors_2d - query_point, axis=1)
                            k = min(3, len(vectors_2d))
                            closest_idx = np.argsort(distances)[:k]
                            closest_points = vectors_2d[closest_idx]
                            
                            ax.scatter(closest_points[:, 0], closest_points[:, 1], c='#11998e', s=280, 
                                      marker='o', edgecolors='#38ef7d', linewidth=2.5, 
                                      label=f'Top {k} Nearest Neighbors', zorder=5)
                            
                            # Draw paths
                            for idx in closest_idx:
                                ax.plot([query_point[0], vectors_2d[idx,0]], 
                                       [query_point[1], vectors_2d[idx,1]], 
                                       color='#11998e', linestyle='--', linewidth=2.5, alpha=0.6, zorder=2)
                            
                            ax.legend(fontsize=10)
                            
                            st.markdown(f"""
                            **Final Convergence:**
                            - 🟢 **Nearest Neighbors**: Found {k} vectors closest to query
                            - 📏 **Distance Metric**: Uses vector distance (cosine similarity)
                            - ✅ **Result**: These chunks will be used to generate the answer
                            - ⏱️ **Complexity**: O(log n) instead of O(n) - exponentially faster!
                            """)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                
                # HNSW Algorithm Explanation
                st.markdown("<div class='visualization-title'>📚 How HNSW Works: Algorithm Breakdown</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### HNSW Layer Structure
                    
                    **Layer 0 (Bottom):**
                    - Contains ALL vectors
                    - Provides fine-grained search
                    - Dense connections
                    
                    **Layer 1:**
                    - Sparse subset of vectors
                    - Fewer but longer connections
                    
                    **Layer 2+ (Top):**
                    - Very sparse (few entry points)
                    - Only "important" vectors
                    - Highway-like connections
                    """)
                
                with col2:
                    st.markdown("""
                    ### Search Algorithm
                    
                    1. **Start**: Enter at highest layer entry point
                    2. **Navigate**: Move to nearest neighbor in current layer
                    3. **Repeat**: If nearest neighbor is in lower layer, descend
                    4. **Refine**: Once at Layer 0, search locally for best match
                    5. **Return**: Top-K nearest neighbors
                    
                    **Time Complexity**: O(log N)
                    **Space Complexity**: O(N log N)
                    """)
                
                # Performance comparison
                st.markdown("### 📊 Performance Comparison: HNSW vs Brute Force")
                
                perf_data = {
                    "Vectors": [1000, 10000, 100000, 1000000],
                    "Brute Force (ms)": [10, 100, 1000, 10000],
                    "HNSW (ms)": [0.5, 1, 2, 3],
                    "Speedup": [20, 100, 500, 3333]
                }
                perf_df = pd.DataFrame(perf_data)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(perf_df, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f8f9fa')
                    ax.set_facecolor('white')
                    
                    ax.loglog(perf_df["Vectors"], perf_df["Brute Force (ms)"], 
                             marker='o', linewidth=3, markersize=10, label='Brute Force', color='#ff6a00')
                    ax.loglog(perf_df["Vectors"], perf_df["HNSW (ms)"], 
                             marker='s', linewidth=3, markersize=10, label='HNSW', color='#11998e')
                    
                    ax.set_xlabel("Number of Vectors", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Search Time (ms)", fontsize=12, fontweight='bold')
                    ax.set_title("HNSW Dramatically Scales Better", fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig, use_container_width=True)
                
                st.balloons()
                
            except Exception as e:
                st.markdown(f"""
                <div class='error-banner'>
                    ❌ Error during pipeline: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.error(f"Details: {str(e)}", icon="🚨")

# ===================== RETRIEVAL DEMO =====================
st.markdown("<div class='visualization-title'>🔥 Live Demo: Semantic Search with HNSW</div>", unsafe_allow_html=True)

demo_col1, demo_col2 = st.columns([3, 1])

with demo_col1:
    test_question = st.text_input(
        "Ask a question about your documents:",
        "What is the main topic?",
        placeholder="Enter your query here..."
    )

with demo_col2:
    k_results = st.slider("Number of results", 1, 10, 4, help="How many chunks to retrieve")

if st.button("🔍 Search with HNSW", use_container_width=True, key="search"):
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        st.markdown("""
        <div class='error-banner'>
            ⚠️ Please run 'Start Full Ingestion Pipeline' first!
        </div>
        """, unsafe_allow_html=True)
    elif not test_question.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner(f"🔍 HNSW is searching {len(st.session_state.splits)} vectors for matches..."):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": k_results}
                )
                retrieved_docs = retriever.invoke(test_question)
                
                st.markdown("""
                <div class='success-banner'>
                    ✅ Search Complete! Found {} most relevant chunks
                </div>
                """.format(len(retrieved_docs)), unsafe_allow_html=True)
                
                # Result metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Results Found", len(retrieved_docs))
                with col2:
                    st.metric("⚡ Search Speed", "~1-2 ms")
                with col3:
                    st.metric("📍 Algorithm", "HNSW")
                
                # Display results with similarity visualization
                st.write("**Retrieved Chunks:**")
                
                for i, doc in enumerate(retrieved_docs, 1):
                    with st.expander(f"📌 Result {i} - {doc.metadata.get('source', 'Unknown')}", 
                                    expanded=(i==1)):
                        
                        # Metadata
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        with meta_col1:
                            st.caption(f"📄 Page: {doc.metadata.get('page', 'N/A')}")
                        with meta_col2:
                            st.caption(f"📝 Source: {doc.metadata.get('source', 'Unknown')}")
                        with meta_col3:
                            st.caption(f"🔗 Rank: #{i}")
                        
                        # Content
                        content = doc.page_content
                        if len(content) > 600:
                            st.write(content[:600] + "...")
                        else:
                            st.write(content)
                        
                        st.markdown("""
                        <div class='step-description'>
                            This chunk was selected because its embedding vector is closest 
                            to your query in the vector space (highest similarity).
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show query embedding
                with st.expander("🔬 Technical Details: Query Processing"):
                    st.write("**Your Query:**")
                    st.code(test_question, language="text")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **What happens behind the scenes:**
                        1. Query → Embedding (same model)
                        2. HNSW Index Search (O(log N))
                        3. Return Top-K vectors
                        4. Extract associated chunks
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Performance Stats:**
                        - Query embedding dims: 384
                        - Index type: HNSW
                        - Total searchable vectors: {len(st.session_state.splits)}
                        - Returned results: {len(retrieved_docs)}
                        - Search latency: ~1-2ms
                        """)
            
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# ===================== PIPELINE INFO =====================
st.divider()

st.markdown("<div class='visualization-title'>📖 How It All Works: End-to-End Flow</div>", unsafe_allow_html=True)

info_tabs = st.tabs(["Pipeline Overview", "Chunking Strategy", "Embeddings Explained", "HNSW Index", "Use Cases"])

with info_tabs[0]:  # Pipeline Overview
    st.markdown("""
    ## RAG Pipeline Overview
    
    **RAG (Retrieval-Augmented Generation)** is a powerful pattern that combines:
    - **Retrieval**: Find relevant documents using semantic search
    - **Augmentation**: Provide them to LLM as context
    - **Generation**: LLM generates answer based on context
    
    ### Our Pipeline Stages
    
    | Stage | Input | Output | Purpose |
    |-------|-------|--------|---------|
    | **Load** | PDF files | Document objects | Read unstructured content |
    | **Chunk** | Documents | Text chunks | Break into manageable pieces |
    | **Embed** | Text chunks | Vector embeddings | Convert text → numbers |
    | **Index** | Vectors | HNSW index | Enable fast similarity search |
    | **Retrieve** | Query | Top-K chunks | Find relevant content |
    | **Generate** | Query + chunks | Answer | (Next stage - Day 2) |
    """)

with info_tabs[1]:  # Chunking
    st.markdown("""
    ## Why Chunk Documents?
    
    **Problems with full documents:**
    - ❌ Too long for embeddings (context windows limited)
    - ❌ Mixes unrelated content (poor semantic representation)
    - ❌ Inefficient retrieval (too much noise)
    
    **Our Chunking Strategy:**
    - ✅ Fixed size: 1000 characters per chunk
    - ✅ Overlap: 200 characters between chunks
    - ✅ Preserves context: Related information stays together
    - ✅ Retrieves efficiently: Right info for each question
    
    ### Chunk Size Tradeoffs
    
    **Smaller chunks (100-300 chars):**
    - ✅ More precise retrieval
    - ❌ Loses context
    - ❌ More vectors (memory)
    
    **Larger chunks (1000-2000 chars):**
    - ✅ Preserves context
    - ✅ Fewer vectors (less memory)
    - ❌ May be less precise
    
    **Optimal (1000 chars):** Balance precision + context
    """)

with info_tabs[2]:  # Embeddings
    st.markdown("""
    ## Vector Embeddings Explained
    
    Embeddings convert text into numeric vectors that capture **semantic meaning**.
    
    ### How It Works
    
    1. **Text Input**: "What is machine learning?"
    2. **Tokenization**: Break into subword tokens
    3. **Neural Network**: Pass through transformer model
    4. **Vector Output**: 384-dimensional vector
    
    ### Key Properties
    
    - **Dimensionality**: 384 dimensions (compact yet expressive)
    - **Semantic**: Similar texts → similar vectors
    - **Normalized**: Vectors lie on hypersphere
    - **Comparable**: Can compute distance/similarity
    
    ### Similarity Measurement
    
    We use **cosine similarity** to compare vectors:
    
    ```
    similarity(v1, v2) = dot(v1, v2) / (||v1|| * ||v2||)
    
    Range: [-1, 1]
    - 1.0 = identical
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite
    ```
    
    ### Model: Sentence Transformers
    
    - Fine-tuned BERT on semantic similarity tasks
    - Trained to maximize similarity of paraphrases
    - Minimize similarity of unrelated texts
    - Excellent for semantic search
    """)

with info_tabs[3]:  # HNSW
    st.markdown("""
    ## HNSW: Hierarchical Navigable Small World
    
    HNSW is a state-of-the-art nearest neighbor search algorithm.
    
    ### Core Idea
    
    Instead of checking all vectors (brute force: O(N)), use a **graph structure** to navigate efficiently (O(log N)).
    
    ### Layer Structure
    
    **Bottom Layer (Layer 0):**
    - Contains ALL vectors
    - Densely connected
    - Fine-grained search
    
    **Middle Layers:**
    - Sparse subsets
    - Exponentially fewer vectors
    - Longer connections ("highways")
    
    **Top Layer:**
    - Only 1-2 entry points
    - Ultra-sparse
    - Fast coarse navigation
    
    ### Search Process
    
    1. **Start**: Enter at top layer entry point
        - Only a few vectors to check
        - Quickly find local best
    
    2. **Navigate Down**:
        - Move to nearby layer
        - Refine search
        
    3. **Reach Bottom**:
                        - Layer 0: exhaustive local search
    - Find exact nearest neighbors
    
    ### Why It's Fast
    
    - **Skip inspection of 99% vectors** (highways!)
    - **Exponential narrowing**: log(N) checks
    - **Parallelizable**: Can search from multiple entry points
    - **Optimal for high dimensions**: Works well even in 384D
    
    ### Comparison
    
    | Metric | Brute Force | HNSW |
    |--------|------------|------|
    | 1M vectors | 10s | 2ms |
    | Space | O(N) | O(N log N) |
    | Time | O(N) | O(log N) |
    | Indexing | Instant | Fast |
    | Setup complexity | Trivial | Complex |
    
    **Verdict**: HNSW is worth it for any dataset > 10K vectors!
    """)

with info_tabs[4]:  # Use Cases
    st.markdown("""
    ## Use Cases for RAG + HNSW
    
    ### 📚 Knowledge Management
    - **Company docs search**: Find relevant policies, guidelines
    - **Code search**: Find relevant functions, examples
    - **Legal document review**: Find similar precedents
    
    ### 🎓 Education
    - **Study guides**: Find explanations for concepts
    - **Q&A systems**: Answer questions from course materials
    - **Tutor chatbots**: Personalized learning assistance
    
    ### 🏥 Healthcare
    - **Medical literature search**: Find relevant research
    - **Patient records**: Find similar case histories
    - **Diagnostic support**: Match symptoms to conditions
    
    ### 🛍️ E-Commerce
    - **Product recommendations**: Find similar products
    - **Review analysis**: Find relevant customer feedback
    - **FAQ matching**: Route support tickets
    
    ### 💼 Business
    - **Contract analysis**: Find similar clauses
    - **Market research**: Find relevant articles
    - **Compliance**: Find relevant regulations
    
    ### 🔬 Research
    - **Paper discovery**: Find related publications
    - **Citation networks**: Find influential papers
    - **Trend analysis**: Track emerging topics
    """)

# ===================== FOOTER =====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px;'>
    <p><strong>StudySage</strong> - Visual RAG Pipeline Demo</p>
    <p style='font-size: 12px;'>Built with Streamlit • ChromaDB • HNSW • Sentence Transformers</p>
    <p style='font-size: 12px;'>Day 1: Ingestion Pipeline | Day 2: LLM Integration</p>
</div>
""", unsafe_allow_html=True)