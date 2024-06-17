import sys, ollama

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = ollama.embeddings(model="nomic-embed-text:v1.5", prompt=question)["embedding"]
print(len(embeddings))
print(embeddings[:4])
