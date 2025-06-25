from typing import Any, Dict, List, Tuple

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

from common import ollama_base_url as base_url
from common import thinking


def get_additional_kwargs_from_model(
    response: BaseModel, exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.model_dump().items() if k not in exclude}


def get_llm_model(model: str, context_window=3900, **kwargs) -> Ollama:
    return Ollama(
        model=model,
        base_url=base_url,
        thinking=thinking,
        context_window=context_window,
        **kwargs
    )


class NormOllamaEmbedding(OllamaEmbedding):

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        result = self._client.embed(
            model=self.model_name, input=texts, options=self.ollama_additional_kwargs
        )
        return result["embeddings"][0]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embed(
            model=self.model_name, input=prompt, options=self.ollama_additional_kwargs
        )
        return result["embeddings"][0]
