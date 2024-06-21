import os, sys

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

loader = DirectoryLoader("data")
data = loader.load()

text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(data)

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
vectorstore = Chroma.from_documents(
    documents=documents, embedding=OllamaEmbeddings(base_url=base_url, model=model_name)
)
print("-" * 80)
print("embed model:", model_name)

question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
retriever = vectorstore.as_retriever(
    # search_kwargs={"k": 2},
)
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
