from langchain_ollama import ChatOllama

from common import ollama_base_url as base_url, ollama_fc_model as model_name
from common.models import demo_fn_call

fn_call_model = ChatOllama(base_url=base_url, model=model_name)
demo_fn_call(fn_call_model, model_name, with_few_shot=True)
