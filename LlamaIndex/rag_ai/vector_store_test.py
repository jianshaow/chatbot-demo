import chromadb
from llama_index.vector_stores import (
    ChromaVectorStore,
    VectorStoreQuery,
)

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("local")
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
