import httpx
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import ssl_verify


def get_llm(model):
    return OpenAILike(
        api_base=api_base,
        api_key=api_key,
        model=model,
        http_client=httpx.Client(verify=ssl_verify),  # type: ignore
        async_http_client=httpx.AsyncClient(verify=ssl_verify),  # type: ignore
    )


def get_embed_model(model_name):
    return OpenAILikeEmbedding(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        http_client=httpx.Client(verify=ssl_verify),
        async_http_client=httpx.AsyncClient(verify=ssl_verify),
    )
