from common import ollama_fc_model as model
from common.models import demo_fn_call_agent
from common.ollama import get_llm

fn_call_model = get_llm(model)
demo_fn_call_agent(fn_call_model)
