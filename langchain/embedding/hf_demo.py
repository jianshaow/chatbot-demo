import os, sys
from langchain_huggingface import HuggingFaceEmbeddings

model = os.environ.get("HF_MODEL", "BAAI/bge-small-en")
embed_model = HuggingFaceEmbeddings(model_name=model)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print(len(embeddings))
print(embeddings[:4])
