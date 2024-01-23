import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("openai_zh")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("LlamaIndex/data_zh").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

retriever = index.as_retriever()
nodes = retriever.retrieve("杨志是个怎样的人?")
for node in nodes:
    print(node)
