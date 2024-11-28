from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

from common import google_chat_model as model_name

chat_model = Gemini(model_name=model_name, transport="rest")
print("-" * 80)
print("chat model:", model_name)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]

print("-" * 80)
response = chat_model.stream_chat(messages)
for chunk in response:
    print(chunk.delta, end="")
print("\n", "-" * 80, sep="")
