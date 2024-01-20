import chromadb

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
collections = db.list_collections()
for collection in collections:
    print(collection)
    count = collection.count()
    print("record count:", count)
    vectors = collection.peek(1)
    for embedding in vectors["embeddings"]:
        print("embedding dimension:", len(embedding))
