import os, sys
import google.generativeai as genai

genai.configure(transport="rest")

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"

model_name = os.environ.get("GEMINI_EMBED_MODEL", "models/embedding-001")
print("-" * 80)
print("embed model:", model_name)
embeddings = genai.embed_content(
    model=model_name, content=question, task_type="retrieval_document"
)["embedding"]
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
