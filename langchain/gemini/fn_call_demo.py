from langchain_google_genai import ChatGoogleGenerativeAI

from common import gemini_fc_model as model_name
from common.models import demo_fn_call

fn_call_model = ChatGoogleGenerativeAI(model=model_name, transport="rest")
demo_fn_call(fn_call_model, model_name, with_few_shot=True)
