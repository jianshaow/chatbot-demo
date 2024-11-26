import sys, ollama

from common import ollama_embed_model as model_name

print("-" * 80)
print("embed model:", model_name)

input = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embedding = ollama.embed(model=model_name, input=input)["embeddings"][0]

print("-" * 80)
print("dimension:", len(embedding))
print(embedding[:4])
print("-" * 80, sep="")
