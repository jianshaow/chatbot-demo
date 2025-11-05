import chromadb
import rag_config
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

config = rag_config.get_config()

embedding = config.embed_model()
llm = config.chat_model()
print("-" * 80)
print("embed_model:", config.embed_model_name)
print("chat model:", config.chat_model_name)

client = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = client.get_or_create_collection(config.vector_db_collection)
vectorstore = Chroma(
    client=client,
    collection_name=config.vector_db_collection,
    embedding_function=embedding,
)


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "retriever", "Retrieve information to help answer a query."
)
agent = create_agent(llm, [retriever_tool])
question = config.get_question()

messages: list = [
    {"role": "user", "content": question},
]

print("-" * 80)
print("Question:", question, sep="\n")
print("Answer:")
for chunk, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
print("\n", "-" * 80, sep="")
