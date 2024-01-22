from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
embeddings = embed_model.get_text_embedding("杨志是一个怎样的人?")
print(len(embeddings))
print(embeddings[:5])
