import os

import chromadb

from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)

    if collection := get_args(2, None):
        client.delete_collection(collection)
    else:
        print("provide the collection name")
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
