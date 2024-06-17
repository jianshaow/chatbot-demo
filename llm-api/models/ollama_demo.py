import ollama

models = ollama.list()["models"]
# print(models)
for model in models:
    print(model)
