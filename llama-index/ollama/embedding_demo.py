import os, sys
from llama_index.embeddings.ollama import OllamaEmbedding

base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
embed_model = OllamaEmbedding(base_url=base_url, model_name=model_name)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80)
