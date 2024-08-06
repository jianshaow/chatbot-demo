import os, sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_name = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
embed_model = HuggingFaceEmbedding(model_name=model_name)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "地球发动机都安装在哪里？"
embeddings = embed_model.get_text_embedding(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80)
