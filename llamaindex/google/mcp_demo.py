from llama_index.llms.google_genai import GoogleGenAI

from common import google_fc_model as model
from common.models import demo_fn_call_agent
from mcp_tools.context7_client import get_http_tools as get_tools

fn_call_model = GoogleGenAI(model=model)
demo_fn_call_agent(
    fn_call_model, tools=get_tools(), query="what is latest version of spring-boot?"
)
