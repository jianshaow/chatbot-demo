import sys
from langchain_huggingface import HuggingFaceEmbeddings

from common import hf_embed_model as model_name

embed_model = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs={"trust_remote_code": True}
)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.embed_query(question)
print("-" * 80)
print("dimension:", len(embeddings))
print(embeddings[:4])
print("-" * 80, sep="")
