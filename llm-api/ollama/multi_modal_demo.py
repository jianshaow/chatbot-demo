import ollama, requests

from common import ollama_mm_model as model

print("-" * 80)
print("chat model:", model)

image = requests.get(
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"
).content

response = ollama.chat(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Identify the city where this photo was taken.",
            "images": [image],
        }
    ],
)

print(response)

print("-" * 80)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print("\n", "-" * 80, sep="")
