import sys
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
question = len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
embeddings = embed_model.get_text_embedding(question)
print(len(embeddings))
print(embeddings[:4])
