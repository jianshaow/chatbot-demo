import sys
from openai import OpenAI

from common import openai_embed_model as model

client = OpenAI()

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
print("-" * 80)
print("embed model:", model)
embeddings = client.embeddings.create(model=model, input=[question]).data[0].embedding
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
