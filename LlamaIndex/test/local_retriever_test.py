from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader

serviceContext = ServiceContext.from_defaults(embed_model="local", llm=None)
print("embed_model:", serviceContext.embed_model.model_name)
documents = SimpleDirectoryReader("LlamaIndex/data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents,
    service_context=serviceContext,
    show_progress=True,
)

retriever = index.as_retriever()
nodes = retriever.retrieve("What did the author do growing up?")
for node in nodes:
    print(node)
