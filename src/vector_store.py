import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings

CHROMA_DB_ROOT = Path(__file__).parent.parent / "chroma_db"


def _user_path(username=None):
    """Each user gets an isolated Chroma collection so resumes never mix."""
    safe_name = (username or "default").replace("/", "_").replace("\\", "_")
    return str(CHROMA_DB_ROOT / safe_name)


def create_vector_store(docs, embeddings=None, username=None):
    """Create a new vector store for this user, replacing any previous resume data."""
    if embeddings is None:
        embeddings = get_embeddings()

    persist_dir = _user_path(username)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # wipe previous resume so old + new never mix
    os.makedirs(persist_dir, exist_ok=True)  # Chroma needs this to already exist

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def get_vectorstore(username=None):
    """Load this user's existing vector store, if any."""
    embeddings = get_embeddings()
    persist_dir = _user_path(username)
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    return None
