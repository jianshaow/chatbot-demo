import os, chromadb

path = os.environ.get("CHROMA_DB_DIR", "LlamaIndex/chroma_db")
db = chromadb.PersistentClient(path=path)
collections = db.list_collections()
print("collections size:", len(collections))
print("===================")
for collection in collections:
    print(collection)
    count = collection.count()
    print("record count:", count)
    vectors = collection.peek(1)
    for embeddings in vectors["embeddings"]:
        print("embeddings dimension:", len(embeddings))
        print(embeddings[:5])
    print("-------------------")
