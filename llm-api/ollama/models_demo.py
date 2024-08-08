import ollama

models = ollama.list()["models"]
print("-" * 80)
for model in models:
    print(model)
print("_" * 80)
