import os, google.generativeai as genai

model_name = os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(model_name)
history = [{"role": "user", "parts": ["You are a pirate with a colorful personality."]}]
chat = model.start_chat(history=history)
response = chat.send_message("What is your name?")

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
