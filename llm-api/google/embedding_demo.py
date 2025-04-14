import google.generativeai as genai

from common import google_embed_model as model
from common.prompts import embed_question as question

print("-" * 80)
print("embed model:", model)

genai.configure(transport="rest")
embeddings = genai.embed_content(
    model=model, content=question, task_type="retrieval_document"
)["embedding"]

print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
