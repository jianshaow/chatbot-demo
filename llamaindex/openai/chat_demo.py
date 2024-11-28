from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from common import openai_chat_model as model

chat_model = OpenAI(model=model)
print("-" * 80)
print("chat model:", chat_model.model)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]

print("-" * 80)
response = chat_model.stream_chat(messages)
for chunk in response:
    print(chunk.delta, end="")
print("\n", "-" * 80, sep="")
