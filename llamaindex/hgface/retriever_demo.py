import sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from common import hf_embed_model as model_name

Settings.embed_model = HuggingFaceEmbedding(model_name, trust_remote_code=True)
print("-" * 80)
print("embed model:", model_name)

documents = SimpleDirectoryReader("data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)

retriever = index.as_retriever(
    similarity_top_k=4,
)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
nodes = retriever.retrieve(question)
for node in nodes:
    print("-" * 80)
    print(node)
