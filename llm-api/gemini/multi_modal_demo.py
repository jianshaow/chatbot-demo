import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai

from common import gemini_chat_model as model_name

genai.configure(transport="rest")

print("-" * 80)
print("multi-modal model:", model_name)

image = Image.open(
    BytesIO(
        requests.get(
            "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"
        ).content
    )
)

model = genai.GenerativeModel(model_name)
response = model.generate_content(
    ["Identify the city where this photo was taken.", image], stream=True
)

print("-" * 80)
for chunk in response:
    print(chunk.text, end="")
print("_" * 80)
