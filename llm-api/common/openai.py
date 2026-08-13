import json
import os

import httpx
from agents import Agent, Runner, RunResult, RunResultStreaming

from common import ssl_verify
from openai import OpenAI


def get_client():
    headers = __get_extra_headers()
    http_client = httpx.Client(verify=ssl_verify, headers=headers)
    return OpenAI(http_client=http_client)


def __get_extra_headers():
    extra_headers_env = os.getenv("EXTRA_HEADERS", "{}")
    return json.loads(extra_headers_env)


def get_agent(*args, **kwargs) -> Agent:
    return Agent(*args, **kwargs)


def run_agent(agent: Agent, prompt: str) -> RunResult:
    return Runner.run_sync(agent, prompt)


def run_streamed_agent(agent: Agent, prompt: str) -> RunResultStreaming:
    return Runner.run_streamed(agent, prompt)
