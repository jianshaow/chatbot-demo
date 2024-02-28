import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("LlamaIndex/data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)
# print("embed_model:", index.service_context.embed_model.model_name)

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
nodes = retriever.retrieve(question)
for node in nodes:
    print(node)
