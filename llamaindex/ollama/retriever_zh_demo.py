from common.models import demo_retrieve
from common.ollama import get_embed_model

model_name = "paraphrase-multilingual:278m"
embed_model = get_embed_model(model_name=model_name)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
