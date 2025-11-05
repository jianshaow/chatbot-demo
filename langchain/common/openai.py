import os

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from common import ssl_verify


def get_chat_model(model):
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    http_async_client = httpx.AsyncClient(verify=ssl_verify, headers=headers)
    return ChatOpenAI(
        model=model, http_client=http_client, http_async_client=http_async_client
    )


def get_embed_model(model):
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    http_async_client = httpx.AsyncClient(verify=ssl_verify, headers=headers)
    return OpenAIEmbeddings(
        model=model, http_client=http_client, http_async_client=http_async_client
    )


def __get_extra_headers():
    extra_headers_env = os.getenv("EXTRA_HEADERS", None)
    header_strs = extra_headers_env.split(",") if extra_headers_env else []
    headers = {}
    for header_str in header_strs:
        key, value = header_str.split(":")
        headers[key] = value
    return headers
