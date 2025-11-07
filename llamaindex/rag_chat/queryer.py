import chromadb
import rag_config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.vector_stores.chroma import ChromaVectorStore

config = rag_config.get_config()

Settings.embed_model = config.embed_model
Settings.llm = config.llm
print("-" * 80)
print("embed model:", config.embed_model_name)
print("chat model:", config.chat_model_name)

client = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = client.get_or_create_collection(config.vector_db_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(streaming=True)
question = config.get_question() or ""
print("-" * 80)
print("Question:", question, sep="\n")
response = query_engine.query(question)
print("Answer:")
if isinstance(response, StreamingResponse):
    response.print_response_stream()
print("\n", "-" * 80, sep="")
