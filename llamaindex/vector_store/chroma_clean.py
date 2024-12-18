import os, chromadb
from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists(db_dir):
    client = chromadb.PersistentClient(path=os.path.join(db_base_dir, db_dir))
    collection = get_args(2, None)
    if collection is None:
        print("provide the collection name")
    else:
        client.delete_collection(collection)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
