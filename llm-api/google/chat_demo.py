from common import google_chat_model as model_name
from common.prompts import chat_question, chat_system
from google import genai
from google.genai.types import GenerateContentConfig

print("-" * 80)
print("chat model:", model_name)

client = genai.Client()
chat = client.chats.create(
    model=model_name, config=GenerateContentConfig(system_instruction=chat_system)
)

response = chat.send_message_stream(message=chat_question)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("\n", "-" * 80, sep="")
