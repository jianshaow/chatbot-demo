import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

PERSIST_DIR = "LlamaIndex/storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("LlamaIndex/data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(streaming=True)
question = "What did the author do growing up?"
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
response.print_response_stream()
print("\n")
