"""Central place to pick which LLM backend answers are generated with. The user can switch
between providers from the sidebar (see app.py) - selection is stored in
st.session_state['llm_provider'] and every chain in the app (chat, hybrid, level, JD, resume
tailor) reads it through get_llm() below, so switching providers changes every tab at once."""
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

PROVIDERS = {
    "Groq (Llama 3.3 70B)": "groq",
    "DeepSeek": "deepseek",
}
DEFAULT_PROVIDER = "groq"


def get_llm(temperature=0.3, max_tokens=1024):
    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER) if _has_session_state() else DEFAULT_PROVIDER

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY missing! Get one at platform.deepseek.com/api_keys "
                "(requires a small account top-up - DeepSeek is not free despite the low "
                "cost) and set it as an environment variable / secret."
            )
        return ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # default: groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing! Check .env file exists at project root")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _has_session_state():
    """Guards against being called somewhere outside a running Streamlit script (e.g. a
    standalone script/test), where st.session_state isn't available."""
    try:
        _ = st.session_state
        return True
    except Exception:
        return False
