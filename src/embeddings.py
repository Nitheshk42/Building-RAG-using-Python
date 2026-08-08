import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@st.cache_resource(show_spinner="Connecting to embedding API...")
def get_embeddings():
    """Calls Hugging Face's hosted Inference API instead of running the model locally.
    Same model (BAAI/bge-small-en-v1.5), same vectors - the only thing that changes is WHERE
    the computation happens. This removes torch/transformers/onnxruntime (~800MB) from the
    app entirely, which was the single biggest contributor to slow cold starts and memory
    pressure on both Render and Cloud Run. Requires HF_API_TOKEN (free, from
    huggingface.co/settings/tokens) to be set as an env var."""
    api_key = os.getenv("HF_API_TOKEN")
    if not api_key:
        raise ValueError(
            "HF_API_TOKEN missing! Get a free token at huggingface.co/settings/tokens "
            "and set it as an environment variable / secret."
        )
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name=MODEL_NAME,
        # langchain_community's default api_url still points at the old
        # api-inference.huggingface.co endpoint, which HF fully decommissioned in late 2025
        # (that's why it was failing DNS resolution, not a local network issue). The router
        # below is the current replacement for non-chat tasks like feature-extraction/embeddings.
        api_url=f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}/pipeline/feature-extraction",
    )

def get_sample_embedding(embeddings, text):
    return embeddings.embed_query(text)
