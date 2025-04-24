from langchain_openai import ChatOpenAI

from common import openai_fc_model as model_name
from common.models import demo_fn_call
from mcp_tools.stdio_client import get_tools

fn_call_model = ChatOpenAI(model=model_name)
demo_fn_call(fn_call_model, model_name, tools=get_tools())
