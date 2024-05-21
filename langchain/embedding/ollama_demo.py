import os, sys
from langchain_community.embeddings.ollama import OllamaEmbeddings

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text:v1.5")
embed_model = OllamaEmbeddings(base_url=base_url, model=model)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print(len(embeddings))
print(embeddings[:4])
