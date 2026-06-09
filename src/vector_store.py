import os
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings

CHROMA_DB_PATH = "chroma_db"

def create_vector_store(docs, embeddings=None):
    """Create new vector store"""
    if embeddings is None:
        embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vectorstore

def get_vectorstore():
    """Load existing vector store"""
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        return Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    return None