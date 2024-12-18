import os, chromadb
from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    collections = client.list_collections()
    print("collections size:", len(collections))
    print("=" * 80)
    for collection in collections:
        print("name:", collection.name)
        count = collection.count()
        print("record count:", count)
        vectors = collection.peek(1)
        for embeddings in vectors["embeddings"]:
            print("embeddings dimension:", len(embeddings))
            print(embeddings[:4])
        print("-" * 80)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
