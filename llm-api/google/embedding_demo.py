from common import google_embed_model as model
from common.prompts import embed_question as question
from google import genai

print("-" * 80)
print("embed model:", model)

client = genai.Client()

embeddings = client.models.embed_content(
    model=model,
    contents=[question],
).embeddings[0].values

print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
