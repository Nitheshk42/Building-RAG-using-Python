import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """Cached across reruns within a running instance - without this, every single call
    (onboarding, vector store create/get, visual pipeline) was reloading the full
    sentence-transformers model into memory from scratch, which is a major chunk of the
    'taking very long' symptom on Render. Docker image should also pre-download the model
    weights at build time (see Dockerfile) so a cold container start doesn't hit the network."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_sample_embedding(embeddings, text):
    return embeddings.embed_query(text)
