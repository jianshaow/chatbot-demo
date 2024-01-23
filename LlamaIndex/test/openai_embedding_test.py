import os
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
embeddings = embed_model.get_text_embedding("What did the author do growing up?")
print(len(embeddings))
