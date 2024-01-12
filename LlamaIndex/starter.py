from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader

from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

from llama_index import set_global_service_context

set_global_service_context(service_context)

documents = SimpleDirectoryReader("data").load_data()
# print([x for x in documents])
index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
