import os, sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_name = os.environ.get("HF_MODEL", "BAAI/bge-small-en")
Settings.embed_model = HuggingFaceEmbedding(model_name)
print("embed_model:", Settings.embed_model.model_name)

documents = SimpleDirectoryReader("data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
nodes = retriever.retrieve(question)
for node in nodes:
    print("---------------------------------------------")
    print(node)
