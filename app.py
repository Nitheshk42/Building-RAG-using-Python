# # # # # # import streamlit as st
# # # # # # import os
# # # # # # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # # # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # from langchain_chroma import Chroma

# # # # # # st.title("🧠 StudySage - Your AI Tutor")
# # # # # # st.write("**Day 1: Smart Document Ingestion with Visual Processing**")

# # # # # # # ===================== SIDEBAR - UPLOAD =====================
# # # # # # with st.sidebar:
# # # # # #     st.header("📤 Upload Documents")
# # # # # #     st.write("Upload your lecture notes / PDFs")
    
# # # # # #     uploaded_files = st.file_uploader(
# # # # # #         "Choose PDF files", 
# # # # # #         type="pdf", 
# # # # # #         accept_multiple_files=True
# # # # # #     )
    
# # # # # #     if uploaded_files:
# # # # # #         st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
# # # # # #         if st.button("💾 Save Files to Data Folder"):
# # # # # #             os.makedirs("data", exist_ok=True)
# # # # # #             for uploaded_file in uploaded_files:
# # # # # #                 file_path = os.path.join("data", uploaded_file.name)
# # # # # #                 with open(file_path, "wb") as f:
# # # # # #                     f.write(uploaded_file.getbuffer())
# # # # # #             st.success(f"✅ All files saved to `data/` folder!")

# # # # # # # ===================== MAIN INGESTION =====================
# # # # # # st.subheader("🚀 Ingest Documents into RAG System")

# # # # # # if st.button("🔄 Ingest & Process Documents", type="primary"):
# # # # # #     if not os.path.exists("data") or len(os.listdir("data")) == 0:
# # # # # #         st.error("❌ No PDFs found in `data/` folder. Please upload first!")
# # # # # #     else:
# # # # # #         with st.spinner("Processing documents..."):
# # # # # #             # Step 1: Load Documents
# # # # # #             st.write("**Step 1: Loading PDFs**")
# # # # # #             loader = PyPDFDirectoryLoader("data")
# # # # # #             documents = loader.load()
# # # # # #             st.success(f"✅ Loaded {len(documents)} document(s)")

# # # # # #             # Step 2: Chunking (with visual)
# # # # # #             st.write("**Step 2: Chunking Documents**")
# # # # # #             text_splitter = RecursiveCharacterTextSplitter(
# # # # # #                 chunk_size=1000,
# # # # # #                 chunk_overlap=200,
# # # # # #                 separators=["\n\n", "\n", ".", "!", "?", " ", ""]
# # # # # #             )
# # # # # #             splits = text_splitter.split_documents(documents)
            
# # # # # #             col1, col2, col3 = st.columns(3)
# # # # # #             with col1:
# # # # # #                 st.metric("Total Chunks Created", len(splits))
# # # # # #             with col2:
# # # # # #                 st.metric("Chunk Size", "1000 chars")
# # # # # #             with col3:
# # # # # #                 st.metric("Overlap", "200 chars")

# # # # # #             # Show sample chunks
# # # # # #             st.write("**Preview of Chunks:**")
# # # # # #             for i in range(min(3, len(splits))):
# # # # # #                 with st.expander(f"Chunk {i+1} - {len(splits[i].page_content)} characters"):
# # # # # #                     st.text_area("Content", splits[i].page_content[:800] + "...", height=200)

# # # # # #             # Step 3: Embeddings
# # # # # #             st.write("**Step 3: Creating Embeddings**")
# # # # # #             embeddings = HuggingFaceEmbeddings(
# # # # # #                 model_name="sentence-transformers/all-MiniLM-L6-v2"
# # # # # #             )
# # # # # #             sample_embedding = embeddings.embed_query("Test sentence")
# # # # # #             st.success(f"✅ Embedding Model Ready (Dimension: {len(sample_embedding)})")

# # # # # #             # Step 4: Vector Store
# # # # # #             st.write("**Step 4: Building Vector Database**")
# # # # # #             vectorstore = Chroma.from_documents(
# # # # # #                 documents=splits,
# # # # # #                 embedding=embeddings,
# # # # # #                 persist_directory="./chroma_db"
# # # # # #             )
# # # # # #             st.success(f"🎉 **Ingestion Complete!** {len(splits)} chunks stored in vector database.")

# # # # # # # Info Section
# # # # # # with st.expander("ℹ️ How This Works (Learning)"):
# # # # # #     st.write("""
# # # # # #     - **Chunk Size 1000**: Each piece is ~1000 characters
# # # # # #     - **Overlap 200**: Helps the model understand context better
# # # # # #     - **Embeddings**: Converts text into numbers (vectors) so AI can search semantically
# # # # # #     """)


# # # # # import streamlit as st
# # # # # import os
# # # # # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # from langchain_chroma import Chroma

# # # # # st.title("🧠 StudySage - Your AI Tutor")
# # # # # st.write("**Day 1: Smart Document Ingestion**")

# # # # # # ===================== SIDEBAR - UPLOAD =====================
# # # # # with st.sidebar:
# # # # #     st.header("📤 Upload Documents")
# # # # #     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
# # # # #     if uploaded_files:
# # # # #         st.success(f"✅ {len(uploaded_files)} file(s) selected")
# # # # #         if st.button("💾 Save Files to Data Folder"):
# # # # #             os.makedirs("data", exist_ok=True)
# # # # #             for uploaded_file in uploaded_files:
# # # # #                 file_path = os.path.join("data", uploaded_file.name)
# # # # #                 with open(file_path, "wb") as f:
# # # # #                     f.write(uploaded_file.getbuffer())
# # # # #             st.success("✅ Files saved successfully!")

# # # # # # ===================== MAIN INGESTION =====================
# # # # # st.subheader("🚀 Ingest Documents into RAG System")

# # # # # if st.button("🔄 Ingest & Process Documents", type="primary"):
# # # # #     if not os.path.exists("data") or len(os.listdir("data")) == 0:
# # # # #         st.error("❌ No PDFs found in `data/` folder. Upload first from sidebar!")
# # # # #     else:
# # # # #         with st.spinner("Processing... This may take 10-30 seconds"):
# # # # #             try:
# # # # #                 # Step 1: Load
# # # # #                 st.write("**1. Loading PDFs...**")
# # # # #                 loader = PyPDFDirectoryLoader("data")
# # # # #                 documents = loader.load()
# # # # #                 st.success(f"✅ Loaded {len(documents)} document(s)")

# # # # #                 # Step 2: Chunking
# # # # #                 st.write("**2. Chunking Documents...**")
# # # # #                 text_splitter = RecursiveCharacterTextSplitter(
# # # # #                     chunk_size=1000, chunk_overlap=200
# # # # #                 )
# # # # #                 splits = text_splitter.split_documents(documents)
                
# # # # #                 col1, col2, col3 = st.columns(3)
# # # # #                 with col1: st.metric("Total Chunks", len(splits))
# # # # #                 with col2: st.metric("Chunk Size", "1000 chars")
# # # # #                 with col3: st.metric("Overlap", "200 chars")

# # # # #                 # Preview
# # # # #                 for i in range(min(2, len(splits))):
# # # # #                     with st.expander(f"Chunk {i+1} Preview"):
# # # # #                         st.text_area("", splits[i].page_content[:700] + "...", height=180)

# # # # #                 # Step 3: Embeddings
# # # # #                 st.write("**3. Creating Embeddings...**")
# # # # #                 embeddings = HuggingFaceEmbeddings(
# # # # #                     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # # # #                 )
# # # # #                 sample_emb = embeddings.embed_query("test")
# # # # #                 st.success(f"✅ Embeddings Ready (Dimension: {len(sample_emb)})")

# # # # #                 # Step 4: Vector Store
# # # # #                 st.write("**4. Building Vector Database...**")
# # # # #                 vectorstore = Chroma.from_documents(
# # # # #                     documents=splits,
# # # # #                     embedding=embeddings,
# # # # #                     persist_directory="./chroma_db"
# # # # #                 )
# # # # #                 st.success(f"🎉 **Success!** {len(splits)} chunks embedded and stored.")

# # # # #             except Exception as e:
# # # # #                 st.error(f"Error: {str(e)}")
# # # # #                 st.info("Tip: Make sure you ran `pip install sentence-transformers`")

# # # # import streamlit as st
# # # # import os
# # # # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # from langchain_chroma import Chroma

# # # # st.title("🧠 StudySage - Your AI Tutor")
# # # # st.write("**Day 1: Full Visual RAG Pipeline**")

# # # # # ===================== SIDEBAR =====================
# # # # with st.sidebar:
# # # #     st.header("📤 Upload Documents")
# # # #     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
# # # #     if uploaded_files:
# # # #         st.success(f"✅ {len(uploaded_files)} file(s) selected")
# # # #         if st.button("💾 Save Files to Data Folder"):
# # # #             os.makedirs("data", exist_ok=True)
# # # #             for uploaded_file in uploaded_files:
# # # #                 file_path = os.path.join("data", uploaded_file.name)
# # # #                 with open(file_path, "wb") as f:
# # # #                     f.write(uploaded_file.getbuffer())
# # # #             st.success("✅ Files saved!")

# # # # # ===================== MAIN PROCESS =====================
# # # # st.subheader("🚀 Ingest Documents - Visual Pipeline")

# # # # if st.button("🔄 Start Full Ingestion", type="primary"):
# # # #     if not os.path.exists("data") or len(os.listdir("data")) == 0:
# # # #         st.error("❌ No PDFs in data folder. Upload first!")
# # # #     else:
# # # #         with st.spinner("Running full pipeline..."):
# # # #             try:
# # # #                 # 1. Load Documents
# # # #                 st.write("**Step 1: Loading PDFs**")
# # # #                 loader = PyPDFDirectoryLoader("data")
# # # #                 documents = loader.load()
# # # #                 st.success(f"✅ Loaded {len(documents)} document(s)")

# # # #                 # 2. Chunking
# # # #                 st.write("**Step 2: Chunking**")
# # # #                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# # # #                 splits = text_splitter.split_documents(documents)
                
# # # #                 col1, col2, col3 = st.columns(3)
# # # #                 with col1: st.metric("Total Chunks", len(splits))
# # # #                 with col2: st.metric("Chunk Size", "1000 chars")
# # # #                 with col3: st.metric("Overlap", "200 chars")

# # # #                 for i in range(min(2, len(splits))):
# # # #                     with st.expander(f"📌 Chunk {i+1} Preview"):
# # # #                         st.text_area("Content:", splits[i].page_content[:600] + "...", height=150)

# # # #                 # 3. EMBEDDINGS - Now with Visual
# # # #                 st.write("**Step 3: Creating Embeddings**")
# # # #                 embeddings = HuggingFaceEmbeddings(
# # # #                     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # # #                 )
                
# # # #                 # Show sample embedding visually
# # # #                 sample_text = splits[0].page_content[:200] if splits else "Sample text"
# # # #                 sample_embedding = embeddings.embed_query(sample_text)
                
# # # #                 st.success(f"✅ Embedding Model: all-MiniLM-L6-v2")
# # # #                 st.write(f"**Vector Dimension:** {len(sample_embedding)}")
                
# # # #                 # Visual representation of embedding
# # # #                 st.write("**First 20 numbers of the Embedding Vector** (for first chunk):")
# # # #                 st.code(sample_embedding[:20], language="python")
                
# # # #                 with st.expander("Why 384 dimensions?"):
# # # #                     st.write("""
# # # #                     - This model converts every chunk into a **list of 384 floating point numbers**.
# # # #                     - Each number represents a learned feature of the text.
# # # #                     - Higher dimension = more detailed semantic understanding.
# # # #                     - 384 is a good balance between accuracy and speed.
# # # #                     """)

# # # #                 # 4. Vector Database
# # # #                 st.write("**Step 4: Storing in Vector Database**")
# # # #                 vectorstore = Chroma.from_documents(
# # # #                     documents=splits,
# # # #                     embedding=embeddings,
# # # #                     persist_directory="./chroma_db"
# # # #                 )
# # # #                 st.success(f"🎉 **Vector Database Created Successfully!**")
# # # #                 st.write(f"Total vectors stored: **{len(splits)}**")

# # # #                 st.balloons()

# # # #             except Exception as e:
# # # #                 st.error(f"Error: {str(e)}")

# # # # # Learning Section
# # # # with st.expander("📘 Understanding Embeddings"):
# # # #     st.write("""
# # # #     **What is happening?**
# # # #     1. Text chunk → Sentence Transformer Model
# # # #     2. Model outputs **384 numbers** (vector)
# # # #     3. Similar meaning texts = vectors are close in space
# # # #     4. This allows semantic search (not just keyword search)
# # # #     """)

# # # import streamlit as st
# # # import os
# # # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # from langchain_chroma import Chroma

# # # st.title("🧠 StudySage - Your AI Tutor")
# # # st.write("**Day 1: Full Visual RAG Pipeline**")

# # # # ===================== SIDEBAR =====================
# # # with st.sidebar:
# # #     st.header("📤 Upload Documents")
# # #     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
# # #     if uploaded_files:
# # #         st.success(f"✅ {len(uploaded_files)} file(s) selected")
# # #         if st.button("💾 Save Files to Data Folder"):
# # #             os.makedirs("data", exist_ok=True)
# # #             for uploaded_file in uploaded_files:
# # #                 file_path = os.path.join("data", uploaded_file.name)
# # #                 with open(file_path, "wb") as f:
# # #                     f.write(uploaded_file.getbuffer())
# # #             st.success("✅ Files saved to data folder!")

# # # # ===================== MAIN INGESTION =====================
# # # st.subheader("🚀 Ingest Documents - Visual Pipeline")

# # # if st.button("🔄 Start Full Ingestion", type="primary"):
# # #     if not os.path.exists("data") or len(os.listdir("data")) == 0:
# # #         st.error("❌ No PDFs found. Upload from sidebar first!")
# # #     else:
# # #         with st.spinner("Running full pipeline..."):
# # #             try:
# # #                 # Step 1: Load Documents
# # #                 st.write("**Step 1: Loading PDFs**")
# # #                 loader = PyPDFDirectoryLoader("data")
# # #                 documents = loader.load()
# # #                 st.success(f"✅ Loaded {len(documents)} document(s)")

# # #                 # Step 2: Chunking - Detailed View
# # #                 st.write("**Step 2: Chunking Documents**")
# # #                 text_splitter = RecursiveCharacterTextSplitter(
# # #                     chunk_size=1000,
# # #                     chunk_overlap=200,
# # #                     separators=["\n\n", "\n", ".", "!", "?", " ", ""]
# # #                 )
# # #                 splits = text_splitter.split_documents(documents)
                
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1: st.metric("Total Chunks", len(splits))
# # #                 with col2: st.metric("Chunk Size", "1000 characters")
# # #                 with col3: st.metric("Chunk Overlap", "200 characters")

# # #                 st.write("**Visual Chunking Examples:**")
# # #                 for i in range(min(3, len(splits))):
# # #                     with st.expander(f"📌 Chunk {i+1} (Length: {len(splits[i].page_content)} chars)"):
# # #                         st.text_area("Content:", splits[i].page_content, height=250)
# # #                         st.caption(f"**Starts with:** {splits[i].page_content[:100]}...")
# # #                         st.caption(f"**Ends with:** ...{splits[i].page_content[-100:]}")
                        
# # #                         if i < len(splits)-1:
# # #                             overlap = splits[i].page_content[-200:]
# # #                             st.info(f"**Overlap with next chunk (200 chars):** {overlap}")

# # #                 # Step 3: Embeddings
# # #                 st.write("**Step 3: Creating Embeddings**")
# # #                 embeddings = HuggingFaceEmbeddings(
# # #                     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # #                 )
                
# # #                 sample_text = splits[0].page_content[:200] if splits else "Sample"
# # #                 sample_embedding = embeddings.embed_query(sample_text)
                
# # #                 st.success(f"✅ Embedding Model: all-MiniLM-L6-v2")
# # #                 st.write(f"**Vector Dimension:** {len(sample_embedding)}")
                
# # #                 st.write("**First 20 numbers of Embedding Vector (for Chunk 1):**")
# # #                 st.code(sample_embedding[:20], language="python")
                
# # #                 with st.expander("Why dimension is 384?"):
# # #                     st.write("""
# # #                     The model turns every text chunk into **384 floating point numbers**. 
# # #                     These numbers capture the meaning of the text. 
# # #                     Similar texts will have vectors that are close to each other.
# # #                     """)

# # #                 # Step 4: Vector Database - Visual
# # #                 st.write("**Step 4: Saving to Vector Database**")
# # #                 st.write("Chroma is saving all chunk embeddings into a local database...")
                
# # #                 vectorstore = Chroma.from_documents(
# # #                     documents=splits,
# # #                     embedding=embeddings,
# # #                     persist_directory="./chroma_db"
# # #                 )
                
# # #                 st.success(f"🎉 **Vector Database Created Successfully!**")
# # #                 st.write(f"**Total Vectors Saved:** {len(splits)}")
                
# # #                 with st.expander("🔍 What happened behind the scenes?"):
# # #                     st.write("""
# # #                     1. Each chunk was converted into a 384-dimensional vector
# # #                     2. All vectors + original text were saved in `./chroma_db` folder
# # #                     3. Chroma created an index for fast similarity search
# # #                     4. Now the app can quickly find relevant chunks when you ask questions
# # #                     """)
                
# # #                 st.balloons()

# # #             except Exception as e:
# # #                 st.error(f"Error: {str(e)}")

# # # # Final Info
# # # with st.expander("📘 Summary of What You Learned Today"):
# # #     st.write("""
# # #     - **Chunking**: Split big documents into small pieces (1000 chars with 200 overlap)
# # #     - **Embedding**: Converted text into numbers (vectors) using sentence-transformers
# # #     - **Vector DB**: Stored all vectors in Chroma for fast retrieval
# # #     """)


# # import streamlit as st
# # import os
# # from src.document_loader import load_documents
# # from src.text_splitter import split_documents
# # from src.embeddings import get_embeddings, get_sample_embedding
# # from src.vector_store import create_vector_store

# # st.title("🧠 StudySage - Your AI Tutor")
# # st.write("**Day 1: Full Visual RAG Pipeline**")

# # # Sidebar Upload
# # with st.sidebar:
# #     st.header("📤 Upload Documents")
# #     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
# #     if uploaded_files:
# #         st.success(f"✅ {len(uploaded_files)} file(s) selected")
# #         if st.button("💾 Save Files to Data Folder"):
# #             os.makedirs("data", exist_ok=True)
# #             for uploaded_file in uploaded_files:
# #                 file_path = os.path.join("data", uploaded_file.name)
# #                 with open(file_path, "wb") as f:
# #                     f.write(uploaded_file.getbuffer())
# #             st.success("✅ Files saved to `data/` folder!")

# # # Main Ingestion
# # st.subheader("🚀 Ingest Documents - Visual Pipeline")

# # if st.button("🔄 Start Full Ingestion", type="primary"):
# #     if not os.path.exists("data") or len(os.listdir("data")) == 0:
# #         st.error("❌ No PDFs found. Upload first!")
# #     else:
# #         with st.spinner("Running full pipeline..."):
# #             try:
# #                 # Step 1
# #                 st.write("**Step 1: Loading PDFs**")
# #                 documents = load_documents()
# #                 st.success(f"✅ Loaded {len(documents)} document(s)")

# #                 # Step 2
# #                 st.write("**Step 2: Chunking Documents**")
# #                 splits = split_documents(documents)
                
# #                 col1, col2, col3 = st.columns(3)
# #                 with col1: st.metric("Total Chunks", len(splits))
# #                 with col2: st.metric("Chunk Size", "1000 characters")
# #                 with col3: st.metric("Chunk Overlap", "200 characters")

# #                 for i in range(min(3, len(splits))):
# #                     with st.expander(f"📌 Chunk {i+1} (Length: {len(splits[i].page_content)} chars)"):
# #                         st.text_area("Content:", splits[i].page_content, height=220)

# #                 # Step 3
# #                 st.write("**Step 3: Creating Embeddings**")
# #                 embeddings = get_embeddings()
# #                 sample_embedding = get_sample_embedding(embeddings, splits[0].page_content[:200])
                
# #                 st.success("✅ Embedding Model Ready")
# #                 st.write(f"**Vector Dimension:** {len(sample_embedding)}")
# #                 st.write("**First 20 numbers of first chunk embedding:**")
# #                 st.code(sample_embedding[:20])

# #                 # Step 4
# #                 st.write("**Step 4: Saving to Vector Database**")
# #                 vectorstore = create_vector_store(splits, embeddings)
                
# #                 st.success("🎉 **Vector Database Created Successfully!**")
# #                 st.write(f"**Total Vectors Stored:** {len(splits)}")

# #                 st.subheader("🔍 What is Happening Behind the Scenes?")
# #                 col1, col2 = st.columns(2)
# #                 with col1:
# #                     st.info("**1. Data Saved in `chroma_db` folder**")
# #                     st.write("• Chunk texts + 384-dim vectors")
# #                 with col2:
# #                     st.info("**2. Chroma Created Index**")
# #                     st.write("• For fast similarity search")
                
# #                 st.success("**3. Ready for Question Answering!**")

# #             except Exception as e:
# #                 st.error(f"Error: {str(e)}")


# import streamlit as st
# import os
# from src.document_loader import load_documents
# from src.text_splitter import split_documents
# from src.embeddings import get_embeddings, get_sample_embedding
# from src.vector_store import create_vector_store, get_vectorstore_info

# st.title("🧠 StudySage - Your AI Tutor")
# st.write("**Day 1: Full Visual RAG Pipeline + ChromaDB Deep Dive**")

# # Sidebar (same as before)
# with st.sidebar:
#     st.header("📤 Upload Documents")
#     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
#     if uploaded_files:
#         st.success(f"✅ {len(uploaded_files)} file(s) selected")
#         if st.button("💾 Save Files to Data Folder"):
#             os.makedirs("data", exist_ok=True)
#             for uploaded_file in uploaded_files:
#                 file_path = os.path.join("data", uploaded_file.name)
#                 with open(file_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
#             st.success("✅ Files saved to `data/` folder!")

# # Main Ingestion
# st.subheader("🚀 Ingest Documents - Visual Pipeline")

# if st.button("🔄 Start Full Ingestion", type="primary"):
#     if not os.path.exists("data") or len(os.listdir("data")) == 0:
#         st.error("❌ No PDFs found. Upload first!")
#     else:
#         with st.spinner("Running full pipeline..."):
#             try:
#                 # Step 1, 2, 3 remain same
#                 st.write("**Step 1: Loading PDFs**")
#                 documents = load_documents()
#                 st.success(f"✅ Loaded {len(documents)} document(s)")

#                 st.write("**Step 2: Chunking Documents**")
#                 splits = split_documents(documents)
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1: st.metric("Total Chunks", len(splits))
#                 with col2: st.metric("Chunk Size", "1000 characters")
#                 with col3: st.metric("Chunk Overlap", "200 characters")

#                 for i in range(min(2, len(splits))):
#                     with st.expander(f"📌 Chunk {i+1} Preview"):
#                         st.text_area("Content:", splits[i].page_content[:500] + "...", height=150)

#                 st.write("**Step 3: Creating Embeddings**")
#                 embeddings = get_embeddings()
#                 sample_embedding = get_sample_embedding(embeddings, splits[0].page_content[:200])
                
#                 st.success("✅ Embedding Model Ready")
#                 st.write(f"**Vector Dimension:** {len(sample_embedding)}")
#                 st.code(sample_embedding[:20])

#                 # ==================== IMPROVED VECTOR DB SECTION ====================
#                 st.write("**Step 4: Saving to Vector Database (ChromaDB)**")
                
#                 vectorstore = create_vector_store(splits, embeddings)
                
#                 st.success("🎉 **Vector Database Created Successfully!**")
                
#                 # Deep Dive into ChromaDB
#                 info = get_vectorstore_info(vectorstore)
                
#                 if info:
#                     st.subheader("🔍 Deep Dive: What ChromaDB Did Behind the Scenes")
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.info("**1. Data Storage**")
#                         st.write(f"• Total Vectors: **{info['total_vectors']}**")
#                         st.write(f"• Dimension: **{info['dimension']}**")
#                         st.write("• Saved in `./chroma_db` folder")
                    
#                     with col2:
#                         st.info("**2. Index Creation**")
#                         st.write(f"• Index Type: **{info['index_type']}**")
#                         st.write("• Built for **fast approximate nearest neighbor search**")
#                         st.write("• Allows quick similarity matching")

#                     st.success("**3. Ready for Retrieval**")
#                     st.write("""
#                     When a user asks a question in the future:
#                     1. Question → Converted to embedding (384 numbers)
#                     2. ChromaDB uses **HNSW Index** to find most similar vectors quickly
#                     3. Returns top relevant chunks (this is called Retrieval)
#                     """)

#                 st.balloons()

#             except Exception as e:
#                 st.error(f"Error: {str(e)}")


import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings, get_sample_embedding
from src.vector_store import create_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter   # Added for safety

st.title("🧠 StudySage - Your AI Tutor")
st.write("**Day 1: Full Visual RAG Pipeline**")

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
        st.error("❌ No PDFs found. Upload first!")
    else:
        with st.spinner("Running pipeline..."):
            try:
                # === Step 1 & 2 & 3 kept exactly same ===
                st.write("**Step 1: Loading PDFs**")
                documents = load_documents()
                st.success(f"✅ Loaded {len(documents)} document(s)")

                st.write("**Step 2: Chunking Documents**")
                splits = split_documents(documents)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Total Chunks", len(splits))
                with col2: st.metric("Chunk Size", "1000 characters")
                with col3: st.metric("Chunk Overlap", "200 characters")

                for i in range(min(3, len(splits))):
                    with st.expander(f"📌 Chunk {i+1} (Length: {len(splits[i].page_content)} chars)"):
                        st.text_area("Content:", splits[i].page_content, height=220)

                st.write("**Step 3: Creating Embeddings**")
                embeddings = get_embeddings()
                sample_embedding = get_sample_embedding(embeddings, splits[0].page_content[:200])
                
                st.success("✅ Embedding Model Ready")
                st.write(f"**Vector Dimension:** {len(sample_embedding)}")
                st.write("**First 20 numbers of first chunk embedding:**")
                st.code(sample_embedding[:20])

                # ==================== VISUAL CHROMA INDEX PART ====================
                st.write("**Step 4: Saving to Vector Database (ChromaDB)**")
                
                vectorstore = create_vector_store(splits, embeddings)
                
                st.success(f"🎉 **Vector Database Created Successfully!**")
                st.write(f"**Total Vectors Stored:** {len(splits)}")

                st.subheader("🔍 Visual: How ChromaDB Uses HNSW Index")

                st.write("**Simulation: How HNSW finds similar vectors quickly**")

                # Simple 2D simulation for visualization
                np.random.seed(42)
                num_points = min(20, len(splits))  # limit for clear plot
                vectors_2d = np.random.rand(num_points, 2) * 10
                
                query_point = np.array([5, 5])  # simulated user query vector
                
                # Calculate distances
                distances = np.linalg.norm(vectors_2d - query_point, axis=1)
                closest_idx = np.argsort(distances)[:3]
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', label='Stored Chunk Vectors', s=100)
                ax.scatter(query_point[0], query_point[1], c='red', label='User Question Vector', s=200, marker='*')
                
                # Highlight closest
                ax.scatter(vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1], 
                          c='green', label='Top 3 Closest Chunks', s=150, edgecolors='black')
                
                ax.set_title("HNSW Index - Finding Nearest Neighbors")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.legend()
                st.pyplot(fig)

                st.write("**How HNSW Works Behind the Scenes:**")
                st.write("""
                1. All chunk vectors are placed in multi-dimensional space
                2. HNSW builds **layers of connections** (like highways + small roads)
                3. When a question comes → it starts from top layer and quickly narrows down
                4. Finds closest vectors **without checking every single chunk**
                """)

                st.info("In real ChromaDB → This process is much faster than brute force search!")

            except Exception as e:
                st.error(f"Error: {str(e)}")