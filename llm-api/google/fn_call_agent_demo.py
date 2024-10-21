import google.generativeai as genai

from common import google_fc_model as model_name
from common.functions import multiply, add

genai.configure(transport="rest")

tools = [multiply, add]

model = genai.GenerativeModel(model_name=model_name, tools=tools)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("What is (121 * 3) + 42?")
print("-" * 80)
print(response.text)
