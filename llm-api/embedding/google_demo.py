import sys
import google.generativeai as genai

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = genai.embed_content(
    model="models/embedding-001", content=question, task_type="retrieval_document"
)["embedding"]
print(len(embeddings))
print(embeddings[:4])
