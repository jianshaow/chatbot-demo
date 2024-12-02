import google.generativeai as genai

from common import gemini_chat_model as model_name

genai.configure(transport="rest")

print("-" * 80)
print("chat model:", model_name)
model = genai.GenerativeModel(
    model_name, system_instruction="You are a pirate with a colorful personality."
)
response = model.generate_content("What is your name?", stream=True)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
