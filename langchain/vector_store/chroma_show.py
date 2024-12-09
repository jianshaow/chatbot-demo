import os, sys, chromadb

base_dir = os.getenv("CHROMA_BASE_DIR", "chroma")

db_dir = len(sys.argv) == 2 and sys.argv[1] or None
if db_dir:
    client = chromadb.PersistentClient(path=os.path.join(base_dir, db_dir))
    collections = client.list_collections()
    print("collections size:", len(collections))
    print("=" * 80)
    for collection in collections:
        print("name:", collection.name)
        count = collection.count()
        print("record count:", count)
        vectors = collection.peek(1)
        for embeddings in vectors["embeddings"]:
            print("embeddings dimension:", len(embeddings))
            print(embeddings[:4])
        print("-" * 80)
else:
    for subpath in os.listdir(base_dir):
        print(subpath)
