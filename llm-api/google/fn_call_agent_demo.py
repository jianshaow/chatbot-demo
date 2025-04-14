import google.generativeai as genai

from common import google_fc_model as model_name
from common.functions import fns
from common.prompts import fn_call_question as question
from common.prompts import fn_call_system as system_prompt
from common.prompts import google_examples as examples

genai.configure(transport="rest")

model = genai.GenerativeModel(
    model_name=model_name, tools=fns.values(), system_instruction=system_prompt
)
print("-" * 80)
print("fn call model:", model_name)

chat = model.start_chat(history=examples, enable_automatic_function_calling=True)
response = chat.send_message(question)
print("-" * 80)
print(response.text)
