import sys
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
question = len(sys.argv) == 2 and sys.argv[1] or "杨志是一个怎样的人?"
embeddings = embed_model.get_text_embedding(question)
print(len(embeddings))
print(embeddings[:5])
