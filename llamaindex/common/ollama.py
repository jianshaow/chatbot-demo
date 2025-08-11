from typing import Any, Dict, Tuple

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

from common import ollama_base_url as base_url
from common import thinking


def get_additional_kwargs_from_model(
    response: BaseModel, exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.model_dump().items() if k not in exclude}


def get_llm(model: str, context_window=3900, **kwargs) -> Ollama:
    return Ollama(
        model=model,
        base_url=base_url,
        thinking=thinking,
        context_window=context_window,
        **kwargs
    )


def get_embed_model(model_name: str) -> OllamaEmbedding:
    return OllamaEmbedding(base_url=base_url, model_name=model_name)
