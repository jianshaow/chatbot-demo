import os, sys, chromadb

path = os.environ.get("CHROMA_DB_DIR", "LlamaIndex/chroma_db")
db = chromadb.PersistentClient(path=path)
collection = len(sys.argv) == 2 and sys.argv[1] or None
if collection is None:
    print("provide the collection name")
else:
    db.delete_collection(collection)
