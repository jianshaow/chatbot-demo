from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding()
embeddings = embed_model.get_text_embedding("What did the author do growing up?")
print(len(embeddings))
print(embeddings[:5])