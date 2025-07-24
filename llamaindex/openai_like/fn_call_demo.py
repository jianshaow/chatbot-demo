import httpx
from llama_index.llms.openai_like import OpenAILike

from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import openai_like_fc_model as model
from common import ssl_verify
from common.models import demo_fn_call

fn_call_model = OpenAILike(
    api_base=api_base,
    api_key=api_key,
    model=model,
    http_client=httpx.Client(verify=ssl_verify),  # type: ignore
)
demo_fn_call(fn_call_model, model)
