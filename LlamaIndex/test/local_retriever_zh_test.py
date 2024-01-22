from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader

serviceContext = ServiceContext.from_defaults(
    embed_model="BAAI/bge-large-zh-v1.5", llm=None
)
print("embed_model:", serviceContext.embed_model.model_name)
documents = SimpleDirectoryReader("LlamaIndex/data_zh").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    service_context=serviceContext,
    show_progress=True,
)

retriever = index.as_retriever()
nodes = retriever.retrieve("杨志是一个怎样的人?")
for node in nodes:
    print(node)
