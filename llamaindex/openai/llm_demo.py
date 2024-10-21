from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from common import openai_chat_model as model

llm = OpenAI(model=model)
print("-" * 80)
print("chat model:", llm.model)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]

print("-" * 80)
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
