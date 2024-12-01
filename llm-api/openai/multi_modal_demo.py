from openai import OpenAI

from common import openai_chat_model as model

print("-" * 80)
print("multi-modal model:", model)
client = OpenAI()
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Identify the city where this photo was taken.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
                    },
                },
            ],
        },
    ],
    stream=True,
)

print("-" * 80)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n", "-" * 80, sep="")
