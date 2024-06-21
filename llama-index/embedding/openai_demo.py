import sys
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
print("-" * 80)
print("embed model:", embed_model.model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
