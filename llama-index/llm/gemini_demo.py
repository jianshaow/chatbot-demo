import os
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

model_name = os.environ.get("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash")
llm = Gemini(model_name=model_name)
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
