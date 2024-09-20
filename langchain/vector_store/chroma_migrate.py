import os, sys, chromadb

if len(sys.argv) != 3:
    print("input source and target, format: <collection>@<db_path>")
else:
    src = sys.argv[1]
    tgt = sys.argv[2]
    path = os.environ.get("CHROMA_DB_DIR", "chroma")

    src_list = src.split("@")
    src_name = src_list[0]
    if len(src_list) == 1:
        src_path = path
    else:
        src_path = src_list[1]

    tgt_list = tgt.split("@")
    tgt_name = tgt_list[0]
    if len(tgt_list) == 1:
        tgt_path = path
    else:
        tgt_path = tgt_list[1]

    src_db = chromadb.PersistentClient(path=src_path)
    tgt_db = chromadb.PersistentClient(path=tgt_path)

    src_collection = src_db.get_collection(src_name)
    tgt_collection = tgt_db.get_or_create_collection(tgt_name)
    vectors = src_collection.peek(src_collection.count())
    tgt_collection.add(
        ids=vectors["ids"],
        embeddings=vectors["embeddings"],
        documents=vectors["documents"],
        metadatas=vectors["metadatas"],
        uris=vectors["uris"],
        images=vectors["data"],
    )

    print("data migrated:", tgt_collection.count())
