import os

import chromadb

from common import db_base_dir, get_args

db_dir = get_args(1)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    collections = client.list_collections()
    print("collections size:", len(collections))
    print("=" * 80)
    for collection in collections:
        print("name:", collection.name)
        count = collection.count()
        print("record count:", count)
        result = collection.peek(1)
        for embeddings in result["embeddings"]:  # type: ignore
            print("embeddings dimension:", len(embeddings))
            print(embeddings[:5])
        print("-" * 80)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
