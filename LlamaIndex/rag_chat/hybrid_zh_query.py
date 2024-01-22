import chromadb
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("local_zh")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-large-zh-v1.5",
    llm=OpenAI(api_base="http://localhost:8000/v1", api_key="EMPTY"),
)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine(streaming=True)
question = "杨志是一个怎样的人?"
print("User:", question, sep="\n")
response = query_engine.query(question)
print("AI:")
response.print_response_stream()
print("\n")
