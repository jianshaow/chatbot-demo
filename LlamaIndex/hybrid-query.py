import chromadb
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI

db = chromadb.PersistentClient(path="LlamaIndex/chroma_local")
chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(
    embed_model="local",
    llm=OpenAI(api_base="http://localhost:8000/v1", api_key="EMPTY"),
)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine()
question = "What did the author do growing up?"
print("User:", question)
response = query_engine.query(question)
print("AI:", response)
