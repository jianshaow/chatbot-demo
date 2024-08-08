import os
from openai import OpenAI

client = OpenAI()
model_name = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
print("-" * 80)
print("chat model:", model_name)
stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a pirate with a colorful personality."},
        {"role": "user", "content": "What is your name?"},
    ],
    stream=True,
)

print("-" * 80)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
