import os, sys, ollama

model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
print("-" * 80)
print("embed model:", model_name)
prompt = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = ollama.embeddings(model=model_name, prompt=prompt)["embedding"]
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
