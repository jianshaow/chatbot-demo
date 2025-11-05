from common.hgface import get_embed_model
from common.models import demo_embed

model_name = "BAAI/bge-small-zh-v1.5"
embed_model = get_embed_model(model_name)
demo_embed(embed_model, model_name, "地球发动机都安装在哪里？")
