from common import google_fc_model as model_name
from common import google_few_shoted as few_shoted
from common.functions import fns
from common.prompts import fn_call_question as question
from common.prompts import fn_call_system as system_prompt
from common.prompts import google_examples as examples
from google import genai
from google.genai import types

model_kwargs = {}
messages = []
if few_shoted:
    model_kwargs["system_instruction"] = system_prompt
    messages.extend(examples)

config = types.GenerateContentConfig(tools=list(fns.values()), **model_kwargs)
client = genai.Client()
print("-" * 80)
print("fn call model:", model_name)

chat = client.chats.create(config=config, model=model_name, history=messages)
response = chat.send_message(question)
if (
    response.candidates
    and response.candidates[0].content
    and response.candidates[0].content.parts
):
    print("-" * 80)
    print(response.candidates[0].content.parts[0].text)
