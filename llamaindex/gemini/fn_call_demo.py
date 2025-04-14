from llama_index.llms.google_genai import GoogleGenAI

from common import gemini_fc_model as model
from common.models import demo_fn_call

fn_call_model = GoogleGenAI(model=model, transport="rest")
demo_fn_call(fn_call_model, model)
