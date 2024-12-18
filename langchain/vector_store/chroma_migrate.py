import os, sys, chromadb
from common import db_base_dir

if len(sys.argv) != 3:
    print("input source and target, format: <collection>@<db_dir>")
else:
    src = sys.argv[1]
    tgt = sys.argv[2]

    src_list = src.split("@")
    src_name = src_list[0]
    if len(src_list) == 1:
        print("input source and target, format: <collection>@<db_dir>")
    else:
        src_path = os.path.join(db_base_dir, src_list[1])

    tgt_list = tgt.split("@")
    tgt_name = tgt_list[0]
    if len(tgt_list) == 1:
        print("input source and target, format: <collection>@<db_dir>")
    else:
        tgt_path = os.path.join(db_base_dir, tgt_list[1])

    src_client = chromadb.PersistentClient(path=src_path)
    tgt_client = chromadb.PersistentClient(path=tgt_path)

    src_collection = src_client.get_collection(src_name)
    tgt_collection = tgt_client.get_or_create_collection(tgt_name)

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
