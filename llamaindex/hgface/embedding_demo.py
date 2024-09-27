import os, sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common import hf_embed_model as model_name

embed_model = HuggingFaceEmbedding(model_name=model_name)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80)
