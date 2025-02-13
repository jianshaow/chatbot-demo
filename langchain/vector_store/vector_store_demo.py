import os

import chromadb
from langchain_chroma import Chroma

from common import db_base_dir, get_args

base_dir = os.getenv("CHROMA_BASE_DIR", "chroma")

db_dir = get_args(1, None)
if db_dir and os.path.exists(path := os.path.join(db_base_dir, db_dir)):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        vectors = chroma_collection.peek(1)
        vector = vectors["embeddings"][0]
        result = chroma_collection.query(vector, n_results=2)
        print("-" * 33, "chroma query", "-" * 33)
        for i in range(len(result["ids"][0])):
            print("Text:", result["documents"][0][i][:60])
            print("distance:", result["distances"][0][i])
            print("-" * 80)

        vectorstore = Chroma(
            client=client,
            collection_name=collection,
        )

        docs = vectorstore.similarity_search_by_vector(vector, k=2)
        print("-" * 30, "vector store query", "-" * 30)
        for doc in docs:
            print(doc.page_content[:80])
            print("-" * 80)
    else:
        collections = client.list_collections()
        for collection in collections:
            print(collection)
else:
    for subpath in os.listdir(base_dir):
        print(subpath)
