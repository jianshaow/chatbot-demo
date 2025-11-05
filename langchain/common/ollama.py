from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from common import ollama_base_url as base_url
from common import thinking


def get_chat_model(model: str, **kwargs) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, reasoning=thinking, **kwargs)


def get_embed_model(model: str, **kwargs) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model, base_url=base_url, **kwargs)
