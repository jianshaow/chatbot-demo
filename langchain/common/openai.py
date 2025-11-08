import json
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
    extra_headers_env = os.getenv("EXTRA_HEADERS", "{}")
    return json.loads(extra_headers_env)
