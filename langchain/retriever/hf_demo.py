import os, sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = DirectoryLoader("data")
data = loader.load()

text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(data)

model = os.environ.get("HF_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
vectorstore = Chroma.from_documents(
    documents=documents, embedding=HuggingFaceEmbeddings(model_name=model)
)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
retriever = vectorstore.as_retriever()
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])