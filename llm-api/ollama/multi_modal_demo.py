import requests

from common import ollama_mm_model as model
from common.prompts import mm_image_url, mm_question
from ollama import Client

print("-" * 80)
print("multi-modal model:", model)

client = Client()

image = requests.get(mm_image_url, timeout=10).content

response = client.chat(
    model=model,
    messages=[{"role": "user", "content": mm_question, "images": [image]}],
    stream=True,
)

print("-" * 80)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print("\n", "-" * 80, sep="")
