from llama_index.llms.gemini import Gemini

from common import gemini_chat_model as model
from common.models import demo_fn_call_agent

fn_call_model = Gemini(model=model, transport="rest")
demo_fn_call_agent(fn_call_model, model)
