from common import google_chat_model as model_name
from common.prompts import chat_question, chat_system
from google import genai
from google.genai.types import GenerateContentConfig

print("-" * 80)
print("chat model:", model_name)

client = genai.Client()
response = client.models.generate_content_stream(
    config=GenerateContentConfig(system_instruction=chat_system),
    model=model_name,
    contents=[chat_question],
)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("\n", "-" * 80)
