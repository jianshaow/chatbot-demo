from common import openai_fc_model as model
from common.models import demo_fn_call
from common.openai_like import get_llm
from mcp_tools import get_sse_tools

fn_call_model = get_llm(model=model)
demo_fn_call(fn_call_model, model, tools=get_sse_tools())
