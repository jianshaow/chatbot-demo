import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
model = os.environ.get("OLLAMA_CHAT_MODEL", "vicuna:7b")
llm = Ollama(base_url=base_url, model=model)
print("-" * 80)
print("chat model:", model)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality."),
    ChatMessage(role="user", content="What is your name?"),
]
print("-" * 80)
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
