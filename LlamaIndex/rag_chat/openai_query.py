import sys, chromadb
from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("openai")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

query_engine = index.as_query_engine(streaming=True)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
response.print_response_stream()
print("\n")
