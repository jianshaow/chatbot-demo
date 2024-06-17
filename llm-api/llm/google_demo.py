import os, google.generativeai as genai

model_name = os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(model_name)
response = model.generate_content("Hello, how are you today?", stream=True)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
