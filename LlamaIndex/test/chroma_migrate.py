import sys, chromadb

if len(sys.argv) != 3:
    print("input source and target collection name")
else:
    source = sys.argv[1]
    target = sys.argv[2]
    db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
    source_collection = db.get_collection(source)
    target_collection = db.get_or_create_collection("test")
    vectors = source_collection.peek(source.count())
    target_collection.add(
        ids=vectors["ids"],
        embeddings=vectors["embeddings"],
        documents=vectors["documents"],
        metadatas=vectors["metadatas"],
        uris=vectors["uris"],
        images=vectors["data"],
    )

    print("data migrated:", target.count())
