import httpx
from llama_index.llms.openai_like import OpenAILike

from common import openai_fc_model as model
from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import openai_like_mm_model as model
from common import ssl_verify
from common.images import show_demo_image
from common.models import demo_multi_modal

mm_model = OpenAILike(
    api_base=api_base,
    api_key=api_key,
    model=model,
    http_client=httpx.Client(verify=ssl_verify),  # type: ignore
)

show_demo_image()
demo_multi_modal(mm_model, model)
