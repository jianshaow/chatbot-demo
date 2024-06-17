import os, sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.ollama import OllamaEmbedding

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model_name = os.environ.get("OLLAMA_MODEL", "nomic-embed-text:v1.5")
Settings.embed_model = OllamaEmbedding(base_url=base_url, model_name=model_name)
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
