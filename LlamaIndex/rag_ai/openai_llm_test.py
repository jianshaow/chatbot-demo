from llama_index.llms import OpenAI, ChatMessage

llm = OpenAI(api_base="http://localhost:8000/v1", api_key="EMPTY")
response = llm.complete("hello")
print(response.text)

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
