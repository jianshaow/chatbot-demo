import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

loader = DirectoryLoader("data")
data = loader.load()

text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
retriever = vectorstore.as_retriever()
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
