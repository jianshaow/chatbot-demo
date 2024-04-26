import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model = os.environ.get("OLLAMA_MODEL", "vicuna:13b")
llm = Ollama(base_url=base_url, model=model)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n-----------------------------------")
