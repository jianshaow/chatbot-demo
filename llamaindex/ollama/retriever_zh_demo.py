import sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.ollama import OllamaEmbedding

from common import ollama_base_url as base_url, ollama_embed_model as model_name

Settings.embed_model = OllamaEmbedding(base_url=base_url, model_name=model_name)
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
