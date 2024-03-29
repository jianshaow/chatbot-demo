import os.path, sys
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

DATA_PATH = "data"
PERSIST_PATH = "storage"
if not os.path.exists(PERSIST_PATH):
    documents = SimpleDirectoryReader(DATA_PATH).load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_PATH)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_PATH)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(streaming=True)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
response.print_response_stream()
print("\n")
