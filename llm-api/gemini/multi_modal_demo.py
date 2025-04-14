from io import BytesIO

import google.generativeai as genai
import requests
from PIL import Image

from common import google_chat_model as model_name
from common.prompts import mm_image_url, mm_question

genai.configure(transport="rest")

print("-" * 80)
print("multi-modal model:", model_name)

image = Image.open(BytesIO(requests.get(mm_image_url, timeout=10).content))

model = genai.GenerativeModel(model_name)
response = model.generate_content([mm_question, image], stream=True)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("\n", "_" * 80)
