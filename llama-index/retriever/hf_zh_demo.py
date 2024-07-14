import os, sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_name = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-small-zh")
Settings.embed_model = HuggingFaceEmbedding(model_name)
print("-" * 80)
print("embed model:", model_name)

documents = SimpleDirectoryReader("data_zh").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)

retriever = index.as_retriever(
    similarity_top_k=4,
)
question = len(sys.argv) == 2 and sys.argv[1] or "地球发动机都安装在哪里？"
nodes = retriever.retrieve(question)
for node in nodes:
    print("-" * 80)
    print(node)
