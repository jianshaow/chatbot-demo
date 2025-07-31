import chromadb
import rag_config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

config = rag_config.get_config()

Settings.embed_model = config.embed_model()
Settings.llm = config.chat_model()
print("-" * 80)
print("embed model:", config.embed_model_name)
print("chat model:", config.chat_model_name)

client = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = client.get_or_create_collection(config.vector_db_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

chat_engine = index.as_chat_engine(verbose=True)
question = config.get_question() or ""
print("-" * 80)
print("Question:", question, sep="\n")
print("-" * 80)
response = chat_engine.stream_chat(question)
print("-" * 80)
print("Answer:")
for chunk in response.response_gen:
    print(chunk, end="")
print("\n", "-" * 80, sep="")
