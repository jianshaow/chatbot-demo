import sys, chromadb

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
collection = len(sys.argv) == 2 and sys.argv[1] or None
if collection is None:
    print("provide the collection name")
else:
    db.delete_collection(collection)
