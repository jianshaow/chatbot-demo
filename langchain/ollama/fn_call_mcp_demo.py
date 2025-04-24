from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url
from common import ollama_fc_model as model
from common.models import demo_fn_call
from mcp_tools import get_tools

fn_call_model = ChatOllama(base_url=base_url, model=model)
demo_fn_call(fn_call_model, model, tools=get_tools(), with_few_shot=True)
