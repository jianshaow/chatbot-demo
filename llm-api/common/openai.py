import httpx

from common import ssl_verify
from openai import OpenAI


def get_client():
    client = OpenAI(http_client=httpx.Client(verify=ssl_verify))
    return client
