import os, sys, chromadb

if len(sys.argv) != 3:
    print("input source and target collection name")
else:
    source = sys.argv[1]
    target = sys.argv[2]
    path = os.environ.get("CHROMA_DB_DIR", "chroma_db")
    db = chromadb.PersistentClient(path=path)
    source_collection = db.get_collection(source)
    target_collection = db.get_or_create_collection(target)
    vectors = source_collection.peek(source_collection.count())
    target_collection.add(
        ids=vectors["ids"],
        embeddings=vectors["embeddings"],
        documents=vectors["documents"],
        metadatas=vectors["metadatas"],
        uris=vectors["uris"],
        images=vectors["data"],
    )

    print("data migrated:", target_collection.count())
