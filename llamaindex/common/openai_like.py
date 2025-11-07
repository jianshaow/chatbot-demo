import os

import httpx
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import openai_like_embed_batch_size, openai_like_is_chat_model, ssl_verify


def get_llm(model: str) -> OpenAILike:
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    async_http_client = httpx.AsyncClient(verify=ssl_verify, headers=headers)
    return OpenAILike(
        api_base=api_base,
        api_key=api_key,
        model=model,
        is_chat_model=openai_like_is_chat_model,
        http_client=http_client,  # type: ignore
        async_http_client=async_http_client,  # type: ignore
    )


def get_embed_model(model_name: str) -> OpenAIEmbedding:
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    async_http_client = httpx.AsyncClient(verify=ssl_verify, headers=headers)
    return OpenAIEmbedding(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        embed_batch_size=openai_like_embed_batch_size,
        http_client=http_client,
        async_http_client=async_http_client,
    )


def __get_extra_headers():
    extra_headers_env = os.getenv("EXTRA_HEADERS", None)
    header_strs = extra_headers_env.split(",") if extra_headers_env else []
    headers = {}
    for header_str in header_strs:
        key, value = header_str.split(":")
        headers[key] = value
    return headers
