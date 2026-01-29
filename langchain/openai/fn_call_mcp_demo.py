from common import openai_fc_model as model_name
from common.models import demo_fn_call
from common.openai import get_chat_model
from mcp_tools.calc_client import get_stdio_tools as get_tools

fn_call_model = get_chat_model(model=model_name)
demo_fn_call(fn_call_model, model_name, tools=get_tools())
