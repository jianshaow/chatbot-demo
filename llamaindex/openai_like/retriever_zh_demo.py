import httpx
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from common import openai_like_api_base as api_base
from common import openai_like_api_key as api_key
from common import openai_like_embed_model as model_name
from common import ssl_verify
from common.models import demo_retrieve

embed_model = OpenAILikeEmbedding(
    model_name=model_name,
    api_base=api_base,
    api_key=api_key,
    http_client=httpx.Client(verify=ssl_verify),
)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
