import os
from llama_index.embeddings import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

embed_model = OpenAIEmbedding()
embeddings = embed_model.get_text_embedding("What did the author do growing up?")
print(len(embeddings))
