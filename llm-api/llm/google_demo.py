import os, google.generativeai as genai

model_name = os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(
    model_name, system_instruction="You are a pirate with a colorful personality."
)
response = model.generate_content("What is your name?")

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
