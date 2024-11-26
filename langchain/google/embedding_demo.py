import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from common import google_embed_model as model

embed_model = GoogleGenerativeAIEmbeddings(model=model, transport="rest")
print("-" * 80)
print("embed model:", model)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embedding = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embedding))
print(embedding[:4])
print("-" * 80, sep="")
