import ollama, requests

from common import ollama_mm_model as model
from common.prompts import mm_image_url, mm_question

print("-" * 80)
print("multi-modal model:", model)

image = requests.get(mm_image_url).content

response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": mm_question, "images": [image]}],
    stream=True,
)

print("-" * 80)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
print("\n", "-" * 80, sep="")
