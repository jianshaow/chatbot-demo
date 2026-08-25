import os

import chromadb
from langchain_chroma import Chroma
import textwrap

from common import db_base_dir, get_args

base_dir = os.getenv("CHROMA_BASE_DIR", "chroma")

db_dir = get_args(1, None)
if db_dir and os.path.exists(path := os.path.join(db_base_dir, db_dir)):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        get_result = chroma_collection.peek(1)
        if get_result["embeddings"] is not None:
            first_embedding = get_result["embeddings"][0]
            query_result = chroma_collection.query(first_embedding, n_results=2)
            print("=" * 33, "chroma query", "=" * 33)
            if (
                query_result["ids"] is not None
                and query_result["documents"] is not None
                and query_result["distances"] is not None
            ):
                for node_id, distance, document in zip(
                    query_result["ids"][0],
                    query_result["distances"][0],
                    query_result["documents"][0],
                ):
                    print("Node ID:", node_id)
                    print(textwrap.fill("Text: " + document[:347] + "..."))
                    print("distance:", distance)
                    print("-" * 80)

        vectorstore = Chroma(
            client=client,
            collection_name=collection,
        )

        docs = vectorstore.similarity_search_by_vector(first_embedding, k=2)  # type: ignore
        print("=" * 30, "vector store query", "=" * 30)
        for doc in docs:
            print(textwrap.fill("Text: " + doc.page_content[:347] + "..."))
            print("-" * 80)
    else:
        collections = client.list_collections()
        for collection in collections:
            print(collection.name)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
