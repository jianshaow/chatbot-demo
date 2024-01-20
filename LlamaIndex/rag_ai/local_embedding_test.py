from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    text_instruction="Represent this sentence for searching relevant passages:",
)
embeddings = embed_model.get_text_embedding("What did the author do growing up?")
print(len(embeddings))
