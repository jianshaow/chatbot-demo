import sys
from langchain_ollama.embeddings import OllamaEmbeddings

from common import ollama_base_url as base_url, ollama_embed_model as model

embed_model = OllamaEmbeddings(base_url=base_url, model=model)
print("-" * 80)
print("embed model:", model)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
