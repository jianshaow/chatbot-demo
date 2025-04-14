from common import google_chat_model as model_name
from common.prompts import chat_question, chat_system
from google import genai
from google.genai import types

print("-" * 80)
print("chat model:", model_name)

client = genai.Client()
response = client.models.generate_content_stream(
    config=types.GenerateContentConfig(system_instruction=chat_system),
    model=model_name,
    contents=[chat_question],
)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
