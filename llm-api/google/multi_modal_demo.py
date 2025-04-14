from io import BytesIO

import requests
from PIL import Image

from common import google_chat_model as model_name
from common.prompts import mm_image_url, mm_question
from google import genai

print("-" * 80)
print("multi-modal model:", model_name)

image = Image.open(BytesIO(requests.get(mm_image_url, timeout=10).content))

client = genai.Client()
response = client.models.generate_content_stream(
    model=model_name, contents=[mm_question, image]
)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("\n", "_" * 80)
