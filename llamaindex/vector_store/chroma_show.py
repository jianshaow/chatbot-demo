import os, chromadb

path = os.environ.get("CHROMA_DB_DIR", "chroma")
db = chromadb.PersistentClient(path=path)
collections = db.list_collections()
print("collections size:", len(collections))
print("=" * 80)
for collection in collections:
    print(collection.get_model())
    count = collection.count()
    print("record count:", count)
    vectors = collection.peek(1)
    for embeddings in vectors["embeddings"]:
        print("embeddings dimension:", len(embeddings))
        print(embeddings[:4])
    print("-" * 80)
