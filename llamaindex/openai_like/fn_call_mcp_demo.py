import httpx
from llama_index.llms.openai_like import OpenAILike

from common import openai_fc_model as model
from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import ssl_verify
from common.models import demo_fn_call
from mcp_tools import get_sse_tools as get_tools

fn_call_model = OpenAILike(
    api_base=api_base,
    api_key=api_key,
    model=model,
    http_client=httpx.Client(verify=ssl_verify),  # type: ignore
)
demo_fn_call(fn_call_model, model, tools=get_tools())
