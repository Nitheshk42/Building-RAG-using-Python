import os
import shutil
from pathlib import Path
# langchain_chroma (not langchain_community.vectorstores.Chroma) - the community version uses
# the deprecated chromadb.Client() global-singleton factory, which has a tenant-bootstrap bug
# ("Could not connect to tenant default_tenant") on some local setups. langchain_chroma uses
# chromadb.PersistentClient() directly, the modern per-directory client, avoiding it entirely.
from langchain_chroma import Chroma
from src.embeddings import get_embeddings

# Same reasoning as auth.py: on Cloud Run the local filesystem doesn't survive across
# instances, so per-user vector stores need to live on a persistent mount in production.
# APP_DATA_DIR (set via env var, pointed at a mounted volume) overrides the default local path.
DATA_ROOT = Path(os.getenv("APP_DATA_DIR", str(Path(__file__).parent.parent)))
CHROMA_DB_ROOT = DATA_ROOT / "chroma_db"


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
    # NOTE: intentionally NOT pre-creating persist_dir here. chromadb 0.4.x has a known bug
    # where its tenant-validation step fails with "Could not connect to tenant default_tenant"
    # when handed a directory that already exists but is still empty (no chroma.sqlite3 in it
    # yet). Letting Chroma.from_documents create the directory itself, on first write, avoids
    # that empty-but-existing state entirely.
    os.makedirs(CHROMA_DB_ROOT, exist_ok=True)  # just the shared parent, not the user's leaf dir

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
