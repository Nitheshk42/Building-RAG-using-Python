import streamlit as st
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store

# ============= KID-FRIENDLY EXPLANATION =============
def show_concept_card(title, emoji, explanation):
    """Display colorful concept cards"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">{emoji} {title}</h3>
        <p style="color: white; margin: 5px 0;">{explanation}</p>
    </div>
    """, unsafe_allow_html=True)


def render_hnsw_diagram(n_total, highlight_indices=None, title="HNSW: Multi-Layer Graph Structure (illustrative)"):
    """Draw a layered HNSW graph. If highlight_indices given, mark those bottom-layer
    nodes red (the actual retrieved chunks) and show an entry point star, to visualize a search."""
    cap = max(3, min(n_total, 40))
    layer_sizes = [max(3, cap // 8), max(6, cap // 3), cap]
    layer_y = [2, 1, 0]
    colors = ['#FFD700', '#4FC3F7', '#66BB6A']
    names = ['Layer 2 (highways)', 'Layer 1 (roads)', 'Layer 0 (all cards)']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    positions = []
    for size, y, color, name in zip(layer_sizes, layer_y, colors, names):
        xs = np.linspace(0.5, 9.5, size)
        ax.scatter(xs, [y] * size, s=220, color=color, edgecolor='black', zorder=3,
                   label=f'{name} ({size} shown of {n_total if y == 0 else size})')
        positions.append(xs)
        for i in range(size - 1):
            ax.plot([xs[i], xs[i + 1]], [y, y], color=color, alpha=0.35, linewidth=1, zorder=1)

    for i in range(len(positions[0])):
        ax.plot([positions[0][i], positions[1][i % len(positions[1])]],
                [layer_y[0], layer_y[1]], color='gray', linestyle='--', alpha=0.4, zorder=1)
    for i in range(0, len(positions[1]), 1):
        j = int(i * len(positions[2]) / len(positions[1]))
        ax.plot([positions[1][i], positions[2][j]], [layer_y[1], layer_y[2]],
                color='gray', linestyle='--', alpha=0.25, zorder=1)

    if highlight_indices:
        bottom_xs = positions[2]
        cap_n = len(bottom_xs)
        for idx in highlight_indices:
            mapped = min(int(idx * cap_n / max(n_total, 1)), cap_n - 1)
            ax.scatter([bottom_xs[mapped]], [layer_y[2]], s=420, color='#FF5252',
                       edgecolor='black', zorder=4)
        ax.scatter([positions[0][0]], [layer_y[0]], marker='*', s=500, color='red',
                   edgecolor='black', zorder=5, label='Search entry point')

    ax.set_yticks(layer_y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xticks([])
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)
    fig.tight_layout()
    return fig

def display_visual_pipeline():
    """All visual learning stages - Day 2"""
    st.title("📚 Visual RAG Pipeline - How It Actually Works")
    st.write("🎯 Watch how documents become searchable magic ✨")

    from src.onboarding import user_data_dir
    username = st.session_state.get("auth_user", "default")
    data_dir = user_data_dir(username)

    # Sidebar Upload
    with st.sidebar:
        st.header("📤 Upload Document")
        st.caption("One file only — keeps this demo grounded in a single resume.")
        uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx", "doc"], accept_multiple_files=False)
        uploaded_files = [uploaded_file] if uploaded_file else []

        if uploaded_files:
            st.success(f"✅ {uploaded_file.name} selected")
            if st.button("💾 Save File to Data Folder"):
                if os.path.exists(data_dir):
                    shutil.rmtree(data_dir)  # replace, don't accumulate
                os.makedirs(data_dir, exist_ok=True)
                for uf in uploaded_files:
                    file_path = os.path.join(data_dir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                st.success("✅ File saved!")

    # Stage Selection
    stage = st.radio("Go to stage:", [
        "1️⃣ Chunking", "2️⃣ Embeddings", "3️⃣ Vector DB", "4️⃣ HNSW Search"
    ], horizontal=True)

    # ======================== STAGE 1: CHUNKING ========================
    if stage == "1️⃣ Chunking":
        st.header("Stage 1: Breaking Books into Pieces")
        show_concept_card(
            "What's Chunking?",
            "✂️",
            "Imagine cutting a long book into small cards. Each card has ~1000 characters (like 200 words). This helps the AI find answers faster!"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("📖 See sample document"):
                if os.path.exists(data_dir) and os.listdir(data_dir):
                    docs = load_documents(data_dir)
                    if docs:
                        st.write(docs[0].page_content[:500])

        with col2:
            if st.button("✂️ Chunk Document", use_container_width=True):
                if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
                    st.error("❌ No files uploaded yet")
                else:
                    documents = load_documents(data_dir)
                    splits = split_documents(documents)
                    st.session_state.splits = splits
                    st.session_state.documents = documents
                    
                    st.success(f"✅ Split into {len(splits)} cards!")
                    st.metric("Total Cards Created", len(splits))
        
        if 'splits' in st.session_state:
            st.subheader("📊 How Big Are Each Card?")
            splits = st.session_state.splits
            
            fig, ax = plt.subplots(figsize=(9, 3))
            chunk_lengths = [len(s.page_content) for s in splits[:20]]
            colors = ['#667eea' if i % 2 == 0 else '#764ba2' for i in range(len(chunk_lengths))]
            
            bars = ax.barh(range(len(chunk_lengths)), chunk_lengths, color=colors, edgecolor='black', linewidth=0.8)
            ax.set_xlabel("Characters (like letters in a word)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Card Number", fontsize=12, fontweight='bold')
            ax.set_title("🎴 Size of Each Card (1st 20 cards)", fontweight='bold', fontsize=14)
            ax.axvline(1000, color='red', linestyle='--', linewidth=3, label='Perfect Size = 1000 chars')
            ax.legend(fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}', 
                       va='center', fontsize=9, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
            
            st.write("**First 3 Cards (Examples):**")
            for i in range(min(3, len(splits))):
                with st.expander(f"🎴 Card {i+1} ({len(splits[i].page_content)} characters)"):
                    st.text(splits[i].page_content[:300] + "...")

    # ======================== STAGE 2: EMBEDDINGS ========================
    elif stage == "2️⃣ Embeddings":
        st.header("Stage 2: Transform Cards into Brain-Vectors")
        show_concept_card(
            "What's an Embedding?",
            "🧠",
            "Each card gets converted into 384 magic numbers (a vector). Think of it like: the AI reads the card and assigns it a position in a huge 384-dimensional space. Similar cards stay close together!"
        )
        
        if st.button("🧠 Create Embeddings", use_container_width=True):
            if 'splits' not in st.session_state:
                st.error("❌ Complete Stage 1 first!")
            else:
                splits = st.session_state.splits
                embeddings_model = get_embeddings()
                
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
                col1.metric("🤖 AI Brain", "all-MiniLM-L6-v2")
                col2.metric("📏 Magic Numbers Per Card", "384")
                col3.metric("💾 Total Cards Converted", len(all_embeddings))
                
                st.success(f"✅ Converted {len(all_embeddings)} cards to vectors!")

        if 'all_embeddings' in st.session_state:
            st.subheader("🌌 Vector Space Map (Squashed to 2D)")
            st.info("💡 RED = Similar cards stay together. BLUE = Different cards spread apart.")
            
            all_emb = st.session_state.all_embeddings
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb)-1))
            embeddings_2d = tsne.fit_transform(all_emb)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=range(len(embeddings_2d)), cmap='cool', 
                                s=200, alpha=0.8, edgecolors='black', linewidth=1)
            ax.set_title("🗺️ Vector Space: Similar Cards Cluster Together", fontweight='bold', fontsize=14)
            ax.set_xlabel("Semantic Meaning Axis 1", fontsize=11)
            ax.set_ylabel("Semantic Meaning Axis 2", fontsize=11)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Card Number', fontsize=11)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # ======================== STAGE 3: VECTOR DB ========================
    elif stage == "3️⃣ Vector DB":
        st.header("Stage 3: Store Cards in Smart Library (HNSW)")
        show_concept_card(
            "What's HNSW?",
            "🏗️",
            "HNSW = Hierarchical Navigable Small World. Like a library with MULTIPLE FLOORS of organized cards. Top floor has few cards (highways), bottom floor has all cards (detailed organization)!"
        )
        
        if st.button("💾 Create Vector Database", use_container_width=True):
            if 'splits' not in st.session_state:
                st.error("❌ Complete Stage 1 first!")
            else:
                splits = st.session_state.splits
                embeddings_model = get_embeddings()
                
                vectorstore = create_vector_store(splits, embeddings_model, username=username)
                st.session_state.vectorstore = vectorstore
                st.session_state.embeddings_model = embeddings_model
                
                # Store embeddings for visualization
                all_embeddings = []
                for split in splits:
                    emb = embeddings_model.embed_query(split.page_content[:300])
                    all_embeddings.append(emb)
                st.session_state.all_embeddings = np.array(all_embeddings)
                
                st.success("✅ Vector DB created with HNSW index!")

        if 'vectorstore' in st.session_state:
            st.subheader("🗺️ Visual: The HNSW Layered Graph")
            st.caption("Illustrative structure — top layer has few 'highway' nodes, bottom layer has every card.")
            fig = render_hnsw_diagram(len(st.session_state.splits))
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("📖 What Just Happened When You Clicked \"Create Vector Database\"")
            st.markdown(f"""
            No searching happened yet — this step only **builds the structure** so search can be
            fast later (that part happens in Stage 4). Here's what building it actually did, in order:

            **1. Every one of your {len(st.session_state.splits)} chunks was placed on the bottom row.**
            🟢 Layer 0 (bottom) always holds *every* chunk — nothing is left out, so nothing is ever unreachable.

            **2. A random smaller subset of those chunks was also copied up to Layer 1, and an even smaller subset up to Layer 2.**
            Think of it like a highway system: 🟢 Layer 0 is every local street, 🔵 Layer 1 is main roads,
            🟡 Layer 2 (top) is the highway on-ramps — far fewer nodes, so a search can skip across
            large distances in one hop instead of crawling street-by-street.

            **3. Each node was connected (the lines you see) to its nearest neighbor nodes within its own layer**,
            based on how similar their embeddings are — not their original chunk order. That's why the
            layout looks scattered rather than sequential: closeness in the diagram approximates
            closeness in *meaning*, not position in your resume.

            **4. The dashed gray lines link each higher-layer node down to its counterpart below it**,
            so once Stage 4 runs a real search, it can drop from highway → main road → local street
            instead of ever touching all {len(st.session_state.splits)} chunks directly.

            **The takeaway:** what you're looking at is a map that makes your resume searchable in a
            handful of hops instead of a full scan — built once here, used every time you ask a
            question in the other tabs.
            """)

    # ======================== STAGE 4: HNSW SEARCH ========================
    elif stage == "4️⃣ HNSW Search":
        st.header("Stage 4: Finding Your Answer with HNSW Magic")
        show_concept_card(
            "How Search Works?",
            "🔍",
            "You ask a question → AI converts it to a vector → Searches HNSW like finding a house: start at TOP floor (fast highways), jump down through floors, find closest cards at BOTTOM!"
        )
        
        query_input = st.text_input("❓ Ask your question:", placeholder="What is...?")
        
        if query_input and 'vectorstore' in st.session_state and 'all_embeddings' in st.session_state:
            embeddings_model = st.session_state.embeddings_model
            vectorstore = st.session_state.vectorstore
            all_embeddings = st.session_state.all_embeddings
            splits = st.session_state.splits
            
            # DEBUG: Show what's being searched
            with st.expander("🔍 DEBUG: What's in the Vector DB?"):
                st.write(f"Total chunks: {len(splits)}")
                st.write("**First 3 chunks (sample):**")
                for i in range(min(3, len(splits))):
                    st.write(f"Chunk {i+1}: {splits[i].page_content[:150]}...")
            
            # Get query vector
            query_vector = embeddings_model.embed_query(query_input)
            
            # Semantic search
            retrieved_docs = vectorstore.similarity_search(query_input, k=5)
            
            # KEYWORD FALLBACK: If similarity is low, search by keywords
            query_keywords = query_input.lower().split()
            keyword_matches = []
            
            for i, split in enumerate(splits):
                content_lower = split.page_content.lower()
                matches = sum(1 for kw in query_keywords if kw in content_lower)
                if matches > 0:
                    keyword_matches.append((i, split, matches))
            
            # Sort by keyword match count
            keyword_matches.sort(key=lambda x: x[2], reverse=True)
            
            # Use top 3 from semantic OR keyword search (whichever better)
            if keyword_matches and keyword_matches[0][2] >= 2:
                retrieved_docs = [match[1] for match in keyword_matches[:3]]
                st.info("🔑 Used keyword matching (better for your question type)")
            
            with st.expander("🔍 DEBUG: Retrieved Results"):
                st.write(f"Query: {query_input}")
                st.write(f"Found {len(retrieved_docs)} results")
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**Result {i+1}:** {doc.page_content[:100]}...")
            
            st.success("✅ Found relevant cards!")

            # ===== VISUAL: SEARCH PATH ON THE HNSW GRAPH =====
            st.subheader("🗺️ Visual: Search Path Through the Layers")
            retrieved_indices = []
            for doc in retrieved_docs:
                for i, s in enumerate(splits):
                    if s.page_content == doc.page_content:
                        retrieved_indices.append(i)
                        break
            fig = render_hnsw_diagram(
                len(splits),
                highlight_indices=retrieved_indices,
                title="🔍 Query enters at the star → travels down → red = your top matches"
            )
            st.pyplot(fig)
            plt.close(fig)

            # ===== READING THIS SPECIFIC SEARCH =====
            st.subheader("📖 Reading This Graph")
            st.markdown(f"""
            For your question **"{query_input}"**:

            1. It was converted into a 384-number vector, same as every chunk.
            2. Search started at the ⭐ **entry point** (top layer) and walked down through the dashed connections, layer by layer.
            3. In Layer 0, it followed neighbor links until it found the chunks whose vectors sit closest to your question's vector.
            4. Those chunks are marked 🔴 **red** on the graph above — they're the exact {len(retrieved_indices)} chunks the LLM was given to answer you.

            Nothing outside the red nodes was used to generate your answer below.
            """)

            # ===== RESULTS WITH SIMILARITY SCORES =====
            st.subheader("📌 Your 3 Most Relevant Cards (Ranked by Similarity)")
            st.info("💡 **Similarity Score = How close is this card's meaning to your question?** (0-100%)")
            
            # Calculate proper similarity scores using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = np.array([query_vector]).reshape(1, -1)
            
            # Get embeddings for retrieved docs
            retrieved_embeddings = []
            for doc in retrieved_docs:
                emb = embeddings_model.embed_query(doc.page_content[:300])
                retrieved_embeddings.append(emb)
            
            retrieved_embeddings = np.array(retrieved_embeddings)
            similarities = cosine_similarity(query_vec, retrieved_embeddings)[0]
            
            # Show results with accurate scores
            for i, (doc, sim_score) in enumerate(zip(retrieved_docs, similarities), 1):
                sim_percent = max(0, int(sim_score * 100))  # Convert -1 to 1 scale to 0-100
                
                with st.expander(f"✅ Card #{i} | Match Score: {sim_percent}%", expanded=(i==1)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write("**📄 Content:**")
                        st.write(doc.page_content[:400])
                        if len(doc.page_content) > 400:
                            st.caption("... (content truncated)")
                    with col2:
                        # Show score with visual bar
                        st.metric("Match Score", f"{sim_percent}%")
                        
                        # Visual indicator
                        if sim_percent >= 80:
                            st.success("⭐ Excellent\nMatch!", icon="✅")
                        elif sim_percent >= 60:
                            st.info("👍 Good\nMatch!", icon="ℹ️")
                        elif sim_percent >= 40:
                            st.warning("✓ Okay\nMatch", icon="⚠️")
                        else:
                            st.error("❌ Poor\nMatch")
                        
                        # Progress bar
                        st.progress(sim_score)  # 0-1 scale
            
            # Add explanation of the process
            st.divider()
            st.subheader("🎓 Understanding the Similarity Score")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **What is Similarity Score?**
                - Compares your question's 384D vector with each card's 384D vector
                - Uses **Cosine Similarity** (measures angle between vectors)
                - Range: 0% = Completely Different | 100% = Identical
                
                **Example:**
                - Q: "What is Python?"
                - Card A: "Python is a programming language..." → 85% match ✅
                - Card B: "Snakes live in jungles..." → 15% match ❌
                
                **Why not 100%?**
                - Your question & answer cards use DIFFERENT words
                - But they share SIMILAR semantic meaning
                - That's why 70-85% is actually EXCELLENT match!
                """)
            
            with col2:
                st.markdown("""
                **Score Interpretation:**
                
                🟢 **80-100%** = Excellent Match!
                - Directly answers your question
                - Very relevant content
                
                🔵 **60-79%** = Good Match
                - Related to your question
                - Useful information
                
                🟡 **40-59%** = Okay Match
                - Somewhat related
                - May need other cards
                
                🔴 **0-39%** = Poor Match
                - Weak relevance
                - Not recommended
                """)
            
            # Visual comparison
            st.subheader("📊 Detailed Search Process")
            st.markdown("""
            **Step-by-Step What HNSW Did:**
            
            1. **Your Question Vector** → Converted to 384 magic numbers (red star)
            2. **Entry Point** → Started at gold point on TOP floor (fastest)
            3. **Highway Jumps** → Jumped across space using orange highways (cover distance fast)
            4. **Descend Layers** → Went down to middle floor (blue roads), then bottom floor (green neighbors)
            5. **Find Neighbors** → Found closest 3 cards to your question vector
            6. **Calculate Scores** → Measured exact similarity % using cosine formula
            7. **Rank Results** → Showed top 3 in order (best match first)
            
            **Why These 3 Cards?**
            - VectorDB searched through ALL cards
            - Found the 3 CLOSEST to your question in 384D space
            - Higher % = closer in that space = more relevant!
            """)


if __name__ == "__main__":
    display_visual_pipeline()