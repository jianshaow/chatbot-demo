from llama_index.llms.openai import OpenAI

from common import openai_fc_model as model
from common.models import demo_fn_call_agent
from mcp_tools.context7_client import get_sse_tools as get_tools

fn_call_model = OpenAI(model=model)
demo_fn_call_agent(
    fn_call_model,
    model,
    tools=get_tools(),
    query="what is latest version of spring-boot?",
)
