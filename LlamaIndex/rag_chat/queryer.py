import chromadb, rag_config
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore

config = rag_config.get_config()

db = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = db.get_or_create_collection(config.vector_db_collection)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
service_context = ServiceContext.from_defaults(
    embed_model=config.embedding_model(),
    llm=config.chat_model(),
)
index = VectorStoreIndex.from_vector_store(
    vector_store, service_context=service_context
)

query_engine = index.as_query_engine(streaming=True)
question = config.get_question()
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
response.print_response_stream()
print("\n")
