from common import openai_like_fc_model as model
from common.models import demo_fn_call_agent
from common.openai_like import get_llm

fn_call_model = get_llm(model=model)
demo_fn_call_agent(fn_call_model)
