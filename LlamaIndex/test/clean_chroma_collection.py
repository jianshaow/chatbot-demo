import sys, chromadb

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
collection = len(sys.argv) == 2 and sys.argv[1] or "local"
db.delete_collection(collection)
