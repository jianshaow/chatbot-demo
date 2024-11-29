from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common.models import demo_recieve

model_name = "BAAI/bge-large-zh-v1.5"
embed_model = HuggingFaceEmbedding(model_name, trust_remote_code=True)
demo_recieve(embed_model, model_name, "data_zh", "地球发动机都安装在哪里？")
