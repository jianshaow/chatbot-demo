import google.generativeai as genai

models = genai.list_models()
for model in models:
    print(model)