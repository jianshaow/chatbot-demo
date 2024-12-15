from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model_name
from common.models import demo_fn_call_agent

fn_call_model = OpenAI(model=model_name)
demo_fn_call_agent(fn_call_model, model_name)
