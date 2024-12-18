import os, chromadb
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import db_base_dir, get_args

db_dir = get_args(1, None)
if db_dir and os.path.exists((path := os.path.join(db_base_dir, db_dir))):
    client = chromadb.PersistentClient(path)
    if collection := get_args(2, None):
        chroma_collection = client.get_collection(collection)
        vectors = chroma_collection.peek(1)
        result = chroma_collection.query(vectors["embeddings"][0], n_results=2)
        print("-" * 33, "chroma query", "-" * 33)
        for i in range(len(result["ids"][0])):
            print("Text:", result["documents"][0][i][:60])
            print("distance:", result["distances"][0][i])
            print("-" * 80)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        query = VectorStoreQuery(
            query_embedding=vectors["embeddings"][0].tolist(),
            similarity_top_k=2,
            mode="default",
        )
        print("-" * 30, "vector store query", "-" * 30)
        result = vector_store.query(query)
        for i in range(len(result.ids)):
            print(result.nodes[i])
            print("similarity:", result.similarities[i])
            print("-" * 80)
    else:
        collections = client.list_collections()
        for col in collections:
            print(col.name)
else:
    for subpath in os.listdir(db_base_dir):
        print(subpath)
