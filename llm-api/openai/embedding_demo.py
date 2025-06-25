from common import openai_embed_model as model
from common.openai import get_client
from common.prompts import embed_question as question

print("-" * 80)
print("embed model:", model)

client = get_client()
embeddings = client.embeddings.create(model=model, input=[question]).data[0].embedding

print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
