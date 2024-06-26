import chromadb, rag_config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

config = rag_config.get_config()

Settings.embed_model = config.embed_model()
Settings.llm = config.chat_model()
print("-" * 80)
print("embed model:", Settings.embed_model.model_name)
print("chat model:", Settings.llm.model)

db = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = db.get_or_create_collection(config.vector_db_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(streaming=True)
question = config.get_question()
print("-" * 80)
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
response.print_response_stream()
print("\n", "-" * 80, sep="")
