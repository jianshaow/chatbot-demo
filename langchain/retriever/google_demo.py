import sys
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class ListEmbeddings(GoogleGenerativeAIEmbeddings):

    def embed_documents(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
        titles: Optional[List[str]] = None,
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        docs = super().embed_documents(texts, task_type, titles, output_dimensionality)
        docs = [list(doc) for doc in docs]
        return docs


loader = DirectoryLoader("data")
data = loader.load()

text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=ListEmbeddings(model="models/embedding-001"),
)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
retriever = vectorstore.as_retriever()
docs = retriever.invoke(question)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
