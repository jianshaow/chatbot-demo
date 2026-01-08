from common import ollama_fc_model as model
from common.models import demo_fn_call_agent
from common.ollama import get_llm
from mcp_tools.context7_client import get_http_tools as get_tools

fn_call_model = get_llm(model=model)
demo_fn_call_agent(
    fn_call_model, tools=get_tools(), query="what is latest version of spring-boot?"
)
