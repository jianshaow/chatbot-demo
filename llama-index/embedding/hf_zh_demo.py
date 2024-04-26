import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
question = len(sys.argv) == 2 and sys.argv[1] or "地球发动机都安装在哪里？"
embeddings = embed_model.get_text_embedding(question)
print(len(embeddings))
print(embeddings[:4])
