from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

from common import google_chat_model as model_name

llm = Gemini(model_name=model_name, transport="rest")
print("-" * 80)
print("chat model:", model_name)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
print("-" * 80)
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
