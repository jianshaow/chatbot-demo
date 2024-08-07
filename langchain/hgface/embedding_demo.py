import os, sys
from langchain_huggingface import HuggingFaceEmbeddings

model_name = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-small-en")
embed_model = HuggingFaceEmbeddings(model_name=model_name)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
