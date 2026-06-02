import streamlit as st
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress annoying transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings, get_sample_embedding
from src.vector_store import create_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("🧠 StudySage - Your AI Tutor")
st.write("**Day 1: Full Visual RAG Pipeline + HNSW Deep Dive**")

# Sidebar (unchanged)
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
            st.success("✅ Files saved to `data/` folder!")

# Main Pipeline
st.subheader("🚀 Ingest Documents - Visual Pipeline")

if st.button("🔄 Start Full Ingestion", type="primary"):
    if not os.path.exists("data") or len(os.listdir("data")) == 0:
        st.error("❌ No PDFs found. Upload from sidebar first!")
    else:
        with st.spinner("Running full pipeline..."):
            try:
                # Step 1: Loading PDFs
                st.write("**Step 1: Loading PDFs**")
                documents = load_documents()
                st.success(f"✅ Loaded {len(documents)} document(s)")

                # Step 2: Chunking
                st.write("**Step 2: Chunking Documents**")
                splits = split_documents(documents)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Total Chunks", len(splits))
                with col2: st.metric("Chunk Size", "1000 characters")
                with col3: st.metric("Chunk Overlap", "200 characters")

                for i in range(min(3, len(splits))):
                    with st.expander(f"📌 Chunk {i+1} (Length: {len(splits[i].page_content)} chars)"):
                        st.text_area("Content:", splits[i].page_content, height=220)

                # Step 3: Embeddings
                st.write("**Step 3: Creating Embeddings**")
                embeddings = get_embeddings()
                sample_embedding = get_sample_embedding(embeddings, splits[0].page_content[:200])
                
                st.success("✅ Embedding Model Ready")
                st.write(f"**Vector Dimension:** {len(sample_embedding)}")
                st.write("**First 20 numbers of first chunk embedding:**")
                st.code(sample_embedding[:20])

                # Step 4: Vector Store + HNSW
            #                    # ==================== PLOT-BASED HNSW VISUALIZATION ====================
            #     st.write("**Step 4: Saving to Vector Database (ChromaDB)**")
                
            #     vectorstore = create_vector_store(splits, embeddings)
                
            #     st.success(f"🎉 **Vector Database Created Successfully!**")
            #     st.write(f"**Total Vectors Stored:** {len(splits)}")

            #     st.subheader("🔍 Visual Simulation: How HNSW Index Works")

            #     np.random.seed(42)
            #     num_points = min(12, len(splits))
            #     vectors_2d = np.random.rand(num_points, 2) * 10
            #     query_point = np.array([5.2, 5.8])

            #     stages = [
            #         "1. All Chunk Vectors in Space",
            #         "2. HNSW Builds Layers (Highways + Small Roads)",
            #         "3. Search Starts from Top Layer (Highway)",
            #         "4. Narrows Down to Closest Chunks"
            #     ]

            #     for stage_num, title in enumerate(stages, 1):
            #         st.write(f"**Stage {stage_num}: {title}**")
                    
            #         fig, ax = plt.subplots(figsize=(9, 7))
            #         ax.set_title(title, fontsize=14, pad=20)
            #         ax.set_xlabel("Dimension 1 (Simplified)")
            #         ax.set_ylabel("Dimension 2 (Simplified)")
                    
            #         # All vectors
            #         ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', s=120, label='Chunk Vectors')
            #         ax.scatter(query_point[0], query_point[1], c='red', s=250, marker='*', label='User Question Vector')

            #         if stage_num == 2:   # Highways + Small Roads
            #             # Highways (long connections)
            #             for i in range(0, num_points, 3):
            #                 for j in range(i+4, num_points, 3):
            #                     ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
            #                            [vectors_2d[i,1], vectors_2d[j,1]], 'red', linewidth=3, alpha=0.8)
            #             # Small roads
            #             for i in range(num_points):
            #                 for j in range(i+1, num_points):
            #                     if np.random.rand() > 0.55 and abs(i-j) < 5:
            #                         ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
            #                                [vectors_2d[i,1], vectors_2d[j,1]], 'gray', linewidth=1, alpha=0.6)
            #             st.write("🔴 **Red Lines** = Highways (Long jumps)")
            #             st.write("⚪ **Gray Lines** = Small Roads (Local connections)")

            #         elif stage_num == 3:   # Start from Top Layer
            #             top_layer = vectors_2d[::3]
            #             ax.scatter(top_layer[:, 0], top_layer[:, 1], c='orange', s=180, 
            #                       label='Top Layer (Highways)', edgecolors='black')
            #             st.write("**Why start from Top Layer?**")
            #             st.write("→ Like taking highway first → Makes big jumps quickly")
            #             st.write("→ Eliminates far away chunks very fast")

            #         elif stage_num == 4:   # Narrowing Down
            #             distances = np.linalg.norm(vectors_2d - query_point, axis=1)
            #             closest_idx = np.argsort(distances)[:3]
            #             ax.scatter(vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1], 
            #                       c='lime', s=200, label='Closest Chunks Found', edgecolors='black')
            #             st.write("**Narrowing Down Process:**")
            #             st.write("→ From highway, it enters small roads")
            #             st.write("→ Finds the most similar chunks accurately")

            #         ax.legend()
            #         st.pyplot(fig)

            #     st.info("**HNSW = Smart Navigation (Highway → Local Roads)** → Fast + Accurate Retrieval")
            # except Exception as e:
            #     st.error(f"Error: {str(e)}")
            #     st.balloons()


                            # ==================== CLEAR HNSW VISUALIZATION ====================
                st.write("**Step 4: Saving to Vector Database (ChromaDB)**")
                
                vectorstore = create_vector_store(splits, embeddings)
                
                st.success(f"🎉 **Vector Database Created Successfully!**")
                st.write(f"**Total Vectors Stored:** {len(splits)}")

                st.subheader("🔍 Visual Simulation: How HNSW Index Works")

                np.random.seed(42)
                num_points = min(15, len(splits))
                vectors_2d = np.random.rand(num_points, 2) * 10
                query_point = np.array([5.0, 5.0])

                stages = [
                    "1. All Chunk Vectors in Space",
                    "2. HNSW Builds Layers (Highways + Small Roads)",
                    "3. Search Starts from Top Layer",
                    "4. Narrows Down to Closest Chunks"
                ]

                for stage_num, title in enumerate(stages, 1):
                    st.write(f"**Stage {stage_num}: {title}**")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.set_title(title, fontsize=14, pad=20)
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    
                    # Plot all vectors
                    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', s=130, label='Chunk Vectors')
                    ax.scatter(query_point[0], query_point[1], c='red', s=280, marker='*', label='User Question Vector')

                    if stage_num == 2:  # Highways vs Small Roads
                        # Highways = Long distance connections (Red)
                        for i in range(num_points):
                            for j in range(i+1, num_points):
                                dist = np.linalg.norm(vectors_2d[i] - vectors_2d[j])
                                if dist > 6:   # Far away points = Highway
                                    ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
                                           [vectors_2d[i,1], vectors_2d[j,1]], 
                                           'red', linewidth=3, alpha=0.7)
                        
                        # Small Roads = Nearby connections (Gray)
                        for i in range(num_points):
                            for j in range(i+1, num_points):
                                dist = np.linalg.norm(vectors_2d[i] - vectors_2d[j])
                                if dist < 3:   # Close points = Small Road
                                    ax.plot([vectors_2d[i,0], vectors_2d[j,0]], 
                                           [vectors_2d[i,1], vectors_2d[j,1]], 
                                           'gray', linewidth=1.2, alpha=0.6)
                        
                        st.write("🔴 **Red Lines** = Highways (connect far away vectors)")
                        st.write("⚪ **Gray Lines** = Small Roads (connect nearby vectors)")

                    elif stage_num == 3:  # Top Layer
                        top_layer_idx = [0, 4, 8, 12][:num_points//3]
                        top_layer = vectors_2d[top_layer_idx]
                        ax.scatter(top_layer[:, 0], top_layer[:, 1], c='orange', s=200, 
                                  label='Top Layer (Highways)', edgecolors='black', linewidth=2)
                        st.write("**Top Layer = Entry Points**")
                        st.write("→ These are important vectors connected by highways")
                        st.write("→ Search **starts here** for fast navigation")

                    elif stage_num == 4:  # Narrowing Down
                        distances = np.linalg.norm(vectors_2d - query_point, axis=1)
                        closest_idx = np.argsort(distances)[:3]
                        ax.scatter(vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1], 
                                  c='lime', s=220, label='Closest Chunks', edgecolors='black')
                        
                        # Draw path from query to closest
                        for idx in closest_idx:
                            ax.plot([query_point[0], vectors_2d[idx,0]], 
                                   [query_point[1], vectors_2d[idx,1]], 
                                   'green', linestyle='--', linewidth=2, alpha=0.8)
                        
                        st.write("**Narrowing Down Process:**")
                        st.write("→ Starts from top layer → follows connections")
                        st.write("→ Finds chunks most similar to question (smallest distance)")

                    ax.legend()
                    st.pyplot(fig)

                st.info("**Summary**: HNSW uses long highways for speed + small roads for accuracy → Very fast retrieval!")
            except Exception as e:
                 st.error(f"Error: {str(e)}")

            st.balloons()