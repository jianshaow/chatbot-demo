from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

llm = Ollama(base_url="http://host.docker.internal:11434", model="vicuna:13b-v1.5-q4_0")

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
print("\n-----------------------------------")
