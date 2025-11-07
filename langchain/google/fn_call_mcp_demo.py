from common import google_fc_model as model_name
from common.google import get_chat_model
from common.models import demo_fn_call
from mcp_tools import get_sse_tools as get_tools

fn_call_model = get_chat_model(model_name)
demo_fn_call(fn_call_model, model_name, tools=get_tools(), with_few_shot=True)
