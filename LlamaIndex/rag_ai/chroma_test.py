import chromadb

db = chromadb.PersistentClient(path="LlamaIndex/chroma_db")
chroma_collection = db.get_or_create_collection("openai")

collections = db.list_collections()
print(collections)
count = chroma_collection.count()
print(count)
vectors = chroma_collection.peek(2)
for embedding in vectors["embeddings"]:
    print(len(embedding))
