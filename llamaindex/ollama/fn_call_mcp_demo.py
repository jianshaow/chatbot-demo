from common import ollama_fc_model as model
from common.models import demo_fn_call
from common.ollama import get_llm
from mcp_tools.calc_client import get_stdio_tools as get_tools

fn_call_model = get_llm(model=model)
demo_fn_call(fn_call_model, tools=get_tools(), with_few_shot=True)
