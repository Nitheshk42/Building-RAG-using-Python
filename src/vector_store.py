# from langchain_chroma import Chroma

# def create_vector_store(splits, embeddings):
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db"
#     )
#     return vectorstore


# def explain_chroma_index():
#     """Returns explanation text for UI"""
#     return """
#     **How ChromaDB Creates Index Behind the Scenes:**

#     1. **Converts all chunks into vectors** (384 numbers each)
#     2. **Builds HNSW Index** (Hierarchical Navigable Small World)
#        - Creates multiple layers of connections between vectors
#        - Like building a smart GPS map of your data
#     3. Saves everything in `./chroma_db` folder:
#        - vector data
#        - metadata (original text)
#        - index files for fast search
#     """

from langchain_chroma import Chroma
import numpy as np

def create_vector_store(splits, embeddings):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore