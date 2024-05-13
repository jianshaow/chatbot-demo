import sys
from langchain_openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings()
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print(len(embeddings))
print(embeddings[:4])
