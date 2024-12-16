from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common.models import demo_embed

model_name = "BAAI/bge-large-zh-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
demo_embed(embed_model, model_name, "地球发动机都安装在哪里？")
