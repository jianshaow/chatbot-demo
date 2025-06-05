import os

import httpx

from openai import OpenAI


def get_client():
    ssl_verify = os.getenv("SSL_VERIFY", "true") == "false"
    client = OpenAI(http_client=httpx.Client(verify=ssl_verify))
    return client
