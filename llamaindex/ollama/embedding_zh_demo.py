from common import ollama_base_url as base_url
from common.models import demo_embed
from common.ollama import get_embed_model

model_name = "paraphrase-multilingual:278m"

embed_model = get_embed_model(model_name=model_name)
demo_embed(embed_model, "地球发动机都安装在哪里？")
