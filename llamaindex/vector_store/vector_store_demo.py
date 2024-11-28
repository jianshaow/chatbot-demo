import os, sys, chromadb
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore

base_dir = os.getenv("CHROMA_BASE_DIR", "chroma")

db_dir = len(sys.argv) >= 2 and sys.argv[1] or None
if db_dir:
    db = chromadb.PersistentClient(path=os.path.join(base_dir, db_dir))
    collection = len(sys.argv) == 3 and sys.argv[2] or "default"
    chroma_collection = db.get_collection(collection)

    vectors = chroma_collection.peek(1)

    result = chroma_collection.query(vectors["embeddings"][0], n_results=2)
    print("-" * 33, "chroma query", "-" * 33)
    for i in range(len(result["ids"][0])):
        print("Text:", result["documents"][0][i][:60])
        print("distance:", result["distances"][0][i])
        print("-" * 80)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    query = VectorStoreQuery(
        query_embedding=vectors["embeddings"][0],
        similarity_top_k=2,
        mode="default",
    )
    # print("-" * 30, "vector store query", "-" * 30)
    # result = vector_store.query(query)
    # for i in range(len(result.ids)):
    #     print(result.nodes[i])
    #     print("similarity:", result.similarities[i])
    #     print("-" * 80)

else:
    for subpath in os.listdir(base_dir):
        print(subpath)
