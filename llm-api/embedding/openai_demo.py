import sys
from openai import OpenAI

client = OpenAI()

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = (
    client.embeddings.create(model="text-embedding-ada-002", input=[question])
    .data[0]
    .embedding
)
print(len(embeddings))
print(embeddings[:4])
