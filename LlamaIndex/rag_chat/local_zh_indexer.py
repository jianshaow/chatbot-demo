import sys, chromadb
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("local_zh")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
serviceContext = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-large-zh-v1.5", llm=None
)
print("embed_model:", serviceContext.embed_model.model_name)
if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("LlamaIndex/data_zh").load_data(
        show_progress=True
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=serviceContext,
        show_progress=True,
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        service_context=serviceContext,
    )

retriever = index.as_retriever()
question = len(sys.argv) == 2 and sys.argv[1] or "杨志是一个怎样的人?"
nodes = retriever.retrieve(question)
for node in nodes:
    print(node)
