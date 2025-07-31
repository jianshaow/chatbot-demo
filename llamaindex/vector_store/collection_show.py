import os
import textwrap

import chromadb

from common import db_base_dir, get_args

db_dir = get_args(1)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        count = int(get_args(3, "4") or 0)
        print("total count:", chroma_collection.count())
        print("show top", count)
        result = chroma_collection.peek(count)
        for node_id, embedding, doc in zip(result["ids"], result["embeddings"], result["documents"]):  # type: ignore
            print("-" * 80)
            print("Node ID:", node_id)
            print("Text:", textwrap.fill(doc[:347] + "..."))
            print("Embedding dimension:", len(embedding))
            print("Embedding:", embedding[:5])
        print("-" * 80)
    else:
        collections = client.list_collections()
        for collection in collections:
            print(collection.name)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
