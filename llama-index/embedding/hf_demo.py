import os, sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_name = os.environ.get("HF_MODEL", "BAAI/bge-small-en")
embed_model = HuggingFaceEmbedding(model_name=model_name)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print(len(embeddings))
print(embeddings[:4])
