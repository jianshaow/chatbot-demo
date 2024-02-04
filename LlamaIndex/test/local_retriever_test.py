import sys
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding

serviceContext = ServiceContext.from_defaults(
    embed_model=HuggingFaceEmbedding(), llm=None
)
print("embed_model:", serviceContext.embed_model.model_name)
documents = SimpleDirectoryReader("LlamaIndex/data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    service_context=serviceContext,
    show_progress=True,
)

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
nodes = retriever.retrieve(question)
for node in nodes:
    print(node)
