import os, chromadb
from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        count = int(get_args(3, "4"))
        result = chroma_collection.peek(count)
        print(result["metadatas"][0].keys())
        for doc in result["documents"]:
            width = 0x00 <= doc[0].encode("utf-8")[0] <= 0x7F and 80 or 40
            print("-" * 80)
            print(doc[:width])
            print("......")
            print(doc[-width:])
        print("-" * 80)
    else:
        collections = client.list_collections()
        for col in collections:
            print(col.name)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
