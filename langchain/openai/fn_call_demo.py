from common import openai_fc_model as model_name
from common.models import demo_fn_call
from common.openai import get_llm

fn_call_model = get_llm(model=model_name)
demo_fn_call(fn_call_model, model_name)
