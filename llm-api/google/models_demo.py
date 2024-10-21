from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(transport="rest")

models = genai.list_models()
print("-" * 80)
for model in models:
    print(model)
print("_" * 80)
