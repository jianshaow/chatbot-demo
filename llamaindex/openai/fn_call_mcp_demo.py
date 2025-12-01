from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model
from common.models import demo_fn_call
from mcp_tools.calc_client import get_calc_sse_tools as get_sse_tools

fn_call_model = OpenAI(model=model)
demo_fn_call(fn_call_model, model, tools=get_sse_tools())
