import chromadb, rag_config
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

config = rag_config.get_config()

db = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = db.get_or_create_collection(config.vector_db_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
Settings.embed_model = config.embedding_model()
print("embed_model:", Settings.embed_model.model_name)

if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader(config.data_path).load_data(show_progress=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever()
question = config.get_question()
nodes = retriever.retrieve(question)
for node in nodes:
    print(node)
