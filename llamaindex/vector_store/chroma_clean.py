import os, sys, chromadb

base_dir = os.getenv("CHROMA_BASE_DIR", "chroma")
db_dir = len(sys.argv) >= 2 and sys.argv[1] or None

if db_dir and os.path.exists(db_dir):
    db = chromadb.PersistentClient(path=os.path.join(base_dir, db_dir))
    collection = len(sys.argv) == 3 and sys.argv[2] or None
    if collection is None:
        print("provide the collection name")
    else:
        db.delete_collection(collection)
else:
    for subpath in os.listdir(base_dir):
        print(subpath)
