import os, sys, chromadb
from langchain_chroma import Chroma

path = os.environ.get("CHROMA_DB_DIR", "chroma")
db = chromadb.PersistentClient(path=path)
collection = len(sys.argv) == 2 and sys.argv[1] or "openai"
chroma_collection = db.get_or_create_collection(collection)

vectors = chroma_collection.peek(1)
vector = vectors["embeddings"][0]
result = chroma_collection.query(vector, n_results=2)
print("-" * 33, "chroma query", "-" * 33)
for i in range(len(result["ids"][0])):
    print("Text:", result["documents"][0][i][:60])
    print("distance:", result["distances"][0][i])
    print("-" * 80)

vectorstore = Chroma(
    client=db,
    collection_name=collection,
)

docs = vectorstore.similarity_search_by_vector(vector, k=2)
print("-" * 30, "vector store query", "-" * 30)
for doc in docs:
    print(doc.page_content[:80])
    print("-" * 80)
