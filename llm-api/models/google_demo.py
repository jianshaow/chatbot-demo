import google.generativeai as genai

models = genai.list_models()
print("-" * 80)
for model in models:
    print(model)
print("_" * 80)
