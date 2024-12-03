import google.generativeai as genai

from common import gemini_chat_model as model_name
from common.prompts import chat_system, chat_question

print("-" * 80)
print("chat model:", model_name)

genai.configure(transport="rest")
model = genai.GenerativeModel(model_name, system_instruction=chat_system)
response = model.generate_content(chat_question, stream=True)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
