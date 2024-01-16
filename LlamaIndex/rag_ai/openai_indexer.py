import os
import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

db = chromadb.PersistentClient(path="LlamaIndex/chroma_openai")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("LlamaIndex/data").load_data(show_progress=True)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

retriever = index.as_retriever()
nodes = retriever.retrieve("What did the author do growing up?")
for node in nodes:
    print(node)
