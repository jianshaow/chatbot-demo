from llama_index.llms.google_genai import GoogleGenAI

from common import google_chat_model as model
from common.models import demo_fn_call_agent

fn_call_model = GoogleGenAI(model=model)
demo_fn_call_agent(fn_call_model, model)
