from common import openai_fc_model as model_name
from common.models import demo_harness
from common.openai import get_chat_model

fn_call_model = get_chat_model(model=model_name)
demo_harness(fn_call_model, model_name)
