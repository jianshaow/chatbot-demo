import os
import textwrap

import chromadb

from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        count = int(get_args(3, "4"))
        print("total count:", chroma_collection.count())
        print("show top", count)
        result = chroma_collection.peek(count)
        for doc in result["documents"]:
            print("-" * 80)
            print(textwrap.fill(doc[:347] + "..."))
        print("-" * 80)
    else:
        collections = client.list_collections()
        for collection in collections:
            print(collection.name)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
