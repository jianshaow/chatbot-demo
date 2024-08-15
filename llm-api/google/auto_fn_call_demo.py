import os, google.generativeai as genai

genai.configure(transport="rest")


def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b


tools = [multiply, add]

model_name = os.environ.get("GEMINI_FC_MODEL", "models/gemini-1.5-flash")
model = genai.GenerativeModel(model_name=model_name, tools=tools)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("What is (121 * 3) + 42?")
print(response.text)
