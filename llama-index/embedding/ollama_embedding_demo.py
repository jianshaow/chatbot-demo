import sys
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
    base_url="http://host.docker.internal:11434",
    model_name="znbang/bge:large-en-v1.5-f16",
)
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print(len(embeddings))
print(embeddings[:4])
