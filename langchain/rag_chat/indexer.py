import textwrap

import chromadb
import rag_config
from langchain_chroma import Chroma

from common.vectorstores import parse_documents

config = rag_config.get_config()

client = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = client.get_or_create_collection(config.vector_db_collection)
embedding = config.embed_model
print("-" * 80)
print("embed_model:", config.embed_model_name)

if chroma_collection.count() == 0:
    documents = parse_documents(config.data_dir)
    vectorstore = Chroma.from_documents(
        client=client,
        collection_name=config.vector_db_collection,
        documents=documents,
        embedding=embedding,
    )
else:
    vectorstore = Chroma(
        client=client,
        collection_name=config.vector_db_collection,
        embedding_function=embedding,
    )

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2},
)
question = config.get_question()
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(textwrap.fill(doc.page_content[:347] + "..."))
print("-" * 80, sep="")
