import os
import sys

import chromadb

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

    result = src_collection.peek(src_collection.count())
    tgt_collection.add(
        ids=result["ids"],
        embeddings=result["embeddings"],
        documents=result["documents"],
        metadatas=result["metadatas"],
        uris=result["uris"],
        images=result["data"],  # type: ignore
    )

    print("data migrated:", tgt_collection.count())
