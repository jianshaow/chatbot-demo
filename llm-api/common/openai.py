import os

import httpx

from common import ssl_verify
from openai import OpenAI


def get_client():
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    return OpenAI(http_client=http_client)


def __get_extra_headers():
    extra_headers_env = os.getenv("EXTRA_HEADERS", "")
    header_strs = extra_headers_env.split(",") if extra_headers_env else []
    headers = {}
    for header_str in header_strs:
        key, value = header_str.split(":")
        headers[key] = value
    return headers
