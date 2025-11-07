from common import google_fc_model as model_name
from common.google import get_chat_model
from common.models import demo_fn_call_agent

fn_call_model = get_chat_model(model_name)
demo_fn_call_agent(fn_call_model, model_name)
