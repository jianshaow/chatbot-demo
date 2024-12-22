import ollama

model_name = "paraphrase-multilingual:278m"
print("-" * 80)
print("embed model:", model_name)

embed_question = "地球发动机都安装在哪里？"
embedding = ollama.embed(model=model_name, input=embed_question)["embeddings"][0]

print("-" * 80)
print("dimension:", len(embedding))
print(embedding[:4])
print("-" * 80, sep="")
