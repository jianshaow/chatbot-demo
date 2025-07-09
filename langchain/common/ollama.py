from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url
from common import thinking


def get_llm_model(model: str, **kwargs) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, reasoning=thinking, **kwargs)
