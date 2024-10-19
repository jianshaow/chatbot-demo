import os, google.generativeai as genai

from common.functions import multiply, add

genai.configure(transport="rest")

tools = [multiply, add]

model_name = os.environ.get("GEMINI_FC_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(model_name=model_name, tools=tools)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("What is (121 * 3) + 42?")
print("-" * 80)
print(response.text)
