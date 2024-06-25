import os, sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings

model_name = os.environ.get("GEMINI_EMBED_MODEL", "models/embedding-001")
embed_model = GoogleGenerativeAIEmbeddings(model=model_name, transport="rest")
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
