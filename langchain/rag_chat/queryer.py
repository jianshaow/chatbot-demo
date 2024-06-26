import chromadb, rag_config
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser


config = rag_config.get_config()

embedding = config.embed_model()
llm = config.chat_model()
print("-" * 80)
print("embed_model:", embedding.model)
print("chat model:", llm.model)

db = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = db.get_or_create_collection(config.vector_db_collection)
vectorstore = Chroma(
    client=db,
    collection_name=config.vector_db_collection,
    embedding_function=embedding,
)


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
prompt = hub.pull("rlm/rag-prompt")
passthrough = RunnablePassthrough()
output_parser = StrOutputParser()

rag_chain: RunnableSerializable = (
    {"context": retriever | format_docs, "question": passthrough}
    | prompt
    | llm
    | output_parser
)
question = config.get_question()
print("-" * 80)
print("Question:", question, sep="\n")
print("Answer:")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
