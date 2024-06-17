import os, sys, ollama

model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text:v1.5")
prompt = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = ollama.embeddings(model=model, prompt=prompt)["embedding"]
print(len(embeddings))
print(embeddings[:4])
