import sys
from openai import OpenAI

from common import openai_embed_model as model
from common.prompts import embed_question as question

print("-" * 80)
print("embed model:", model)

client = OpenAI()
embeddings = client.embeddings.create(model=model, input=[question]).data[0].embedding

print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
