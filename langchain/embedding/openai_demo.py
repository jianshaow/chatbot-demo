import sys
from langchain_openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings()
print("-" * 80)
print("embed model:", embed_model.model)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
