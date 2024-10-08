from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

from common import ollama_base_url as base_url, ollama_chat_model as model

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
