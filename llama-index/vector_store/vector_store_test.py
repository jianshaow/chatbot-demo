import os, sys, chromadb
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore

path = os.environ.get("CHROMA_DB_DIR", "chroma")
db = chromadb.PersistentClient(path=path)
collection = len(sys.argv) == 2 and sys.argv[1] or "hface"
chroma_collection = db.get_or_create_collection(collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

vectors = chroma_collection.peek(1)
query = VectorStoreQuery(
    query_embedding=vectors["embeddings"][0],
    similarity_top_k=2,
    mode="default",
)

result = vector_store.query(query)
for i in range(len(result.ids)):
    print(result.nodes[i])
    print("similarity:", result.similarities[i])
