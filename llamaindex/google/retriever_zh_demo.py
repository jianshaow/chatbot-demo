from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from common import google_embed_model as model_name
from common.models import demo_retrieve

embed_model = GoogleGenAIEmbedding(model_name=model_name)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
