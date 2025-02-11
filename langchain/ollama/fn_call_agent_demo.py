from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url, ollama_fc_model as model
from common.models import demo_fn_call_agent

fn_call_model = ChatOllama(base_url=base_url, model=model)
demo_fn_call_agent(fn_call_model, model)
