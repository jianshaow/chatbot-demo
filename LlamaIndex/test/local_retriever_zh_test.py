import sys
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader

serviceContext = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-large-zh-v1.5", llm=None
)
print("embed_model:", serviceContext.embed_model.model_name)
documents = SimpleDirectoryReader("LlamaIndex/data_zh").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    service_context=serviceContext,
    show_progress=True,
)

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "杨志是一个怎样的人?"
nodes = retriever.retrieve(question)
for node in nodes:
    print(node)
