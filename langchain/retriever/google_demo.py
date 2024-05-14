import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

loader = DirectoryLoader("data")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
docs = vectorstore.similarity_search(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
