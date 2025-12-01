from llama_index.llms.google_genai import GoogleGenAI

from common import google_fc_model as model
from common.models import demo_fn_call
from mcp_tools.calc_client import get_calc_stdio_tools as get_tools

fn_call_model = GoogleGenAI(model=model)
demo_fn_call(fn_call_model, model, tools=get_tools())
