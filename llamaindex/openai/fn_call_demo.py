from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model
from common.models import demo_fn_call

fn_call_model = OpenAI(model=model)
demo_fn_call(fn_call_model)
