import os

from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured.document_loaders import UnstructuredLoader


def get_vector_store(
    embed_model: Embeddings, data_path: str = "data/default"
) -> VectorStore:
    vectorstore = Chroma.from_documents(
        documents=parse_documents(data_path),
        embedding=embed_model,
    )
    return vectorstore


def parse_documents(data_path: str) -> list[Document]:
    loader = UnstructuredLoader(
        __list_file(data_path),
        chunking_strategy="basic",
        max_characters=1000000,
        include_orig_elements=False,
    )
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(loader.load())


def __list_file(data_path: str) -> list[str]:
    return [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ]
