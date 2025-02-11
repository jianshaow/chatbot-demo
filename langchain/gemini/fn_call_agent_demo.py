from langchain_google_genai import ChatGoogleGenerativeAI

from common import gemini_fc_model as model
from common.models import demo_fn_call_agent

fn_call_model = ChatGoogleGenerativeAI(model=model, transport="rest")
demo_fn_call_agent(fn_call_model, model)
