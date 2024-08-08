import os, sys
from openai import OpenAI

client = OpenAI()

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
print("-" * 80)
print("embed model:", model_name)
embeddings = (
    client.embeddings.create(model=model_name, input=[question]).data[0].embedding
)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
