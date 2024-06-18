import ollama

models = ollama.list()["models"]
for model in models:
    print(model)
