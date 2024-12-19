from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common.models import demo_retrieve

model_name = "BAAI/bge-small-zh-v1.5"
embed_model = HuggingFaceEmbedding(model_name, trust_remote_code=True)
demo_retrieve(embed_model, model_name, "data/zh-text", "地球发动机都安装在哪里？")
