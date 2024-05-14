import os, sys, chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

path = os.environ.get("CHROMA_DB_DIR", "chroma")
db = chromadb.PersistentClient(path=path)
collection = len(sys.argv) == 2 and sys.argv[1] or "openai"
chroma_collection = db.get_or_create_collection(collection)
embedding = OpenAIEmbeddings()

if chroma_collection.count() == 0:
    print("no vectors in db, load from documents...")
    data = DirectoryLoader("data").load()
    text_splitter = CharacterTextSplitter()
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(
        client=db,
        collection_name=collection,
        documents=all_splits,
        embedding=embedding,
    )
else:
    vectors = chroma_collection.peek(1)
    vector = vectors["embeddings"][0]
    result = chroma_collection.query(vector, n_results=2)
    print("-" * 33, "chroma query", "-" * 33)
    for i in range(len(result["ids"][0])):
        print("Text:", result["documents"][0][i][:60])
        print("distance:", result["distances"][0][i])
        print("-" * 80)

    vectorstore = Chroma(
        client=db, collection_name=collection, embedding_function=embedding
    )

docs = vectorstore.similarity_search_by_vector(vector, k=2)
print("-" * 30, "vector store query", "-" * 30)
for doc in docs:
    print("-" * 80)
    print(doc.page_content[:80])
