from llama_index.llms.gemini import Gemini

from common import gemini_chat_model as model_name
from common.models import demo_fn_call_agent

fn_call_model = Gemini(model_name=model_name, transport="rest")
demo_fn_call_agent(fn_call_model, model_name)
