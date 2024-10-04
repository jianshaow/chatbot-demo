import sys

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from common import ollama_base_url as base_url, ollama_embed_model as model

loader = DirectoryLoader("data")
data = loader.load()

text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=documents, embedding=OllamaEmbeddings(base_url=base_url, model=model)
)
print("-" * 80)
print("embed model:", model)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
retriever = vectorstore.as_retriever(
    # search_kwargs={"k": 2},
)
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
