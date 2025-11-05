from common import ollama_fc_model as model
from common.models import demo_fn_call
from common.ollama import get_chat_model

fn_call_model = get_chat_model(model)
demo_fn_call(fn_call_model, model, with_few_shot=True)
