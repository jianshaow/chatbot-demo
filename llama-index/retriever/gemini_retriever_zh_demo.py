import sys
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.gemini import GeminiEmbedding

Settings.embed_model = GeminiEmbedding()
print("embed_model:", Settings.embed_model.model_name)

documents = SimpleDirectoryReader("data_zh").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "地球发动机都安装在哪里？"
nodes = retriever.retrieve(question)
for node in nodes:
    print("---------------------------------------------")
    print(node)