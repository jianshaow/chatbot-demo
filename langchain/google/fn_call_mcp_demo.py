from langchain_google_genai import ChatGoogleGenerativeAI

from common import google_fc_model as model
from common.models import demo_fn_call
from mcp_tools import get_stdio_tools

fn_call_model = ChatGoogleGenerativeAI(model=model, transport="rest")
demo_fn_call(fn_call_model, model, tools=get_stdio_tools(), with_few_shot=True)
