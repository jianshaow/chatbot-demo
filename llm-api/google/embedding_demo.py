import sys
import google.generativeai as genai

from common import google_embed_model as model

genai.configure(transport="rest")

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"

print("-" * 80)
print("embed model:", model)
embeddings = genai.embed_content(
    model=model, content=question, task_type="retrieval_document"
)["embedding"]
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
