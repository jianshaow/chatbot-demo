import os

from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured.document_loaders import UnstructuredLoader


def get_vector_store(embed_model: Embeddings, data_path: str = "data/default"):
    loader = UnstructuredLoader(
        __list_file(data_path),
        chunking_strategy="basic",
        max_characters=1000000,
        include_orig_elements=False,
    )
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    documents = text_splitter.split_documents(loader.load())
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
    )
    return vectorstore


def __list_file(data_path: str):
    return [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ]
