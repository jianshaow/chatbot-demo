from openai import OpenAI

from common import openai_chat_model as model

print("-" * 80)
print("chat model:", model)
client = OpenAI()
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a pirate with a colorful personality."},
        {"role": "user", "content": "What is your name?"},
    ],
    stream=True,
)

print("-" * 80)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
