import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print(len(embeddings))
print(embeddings[:4])
