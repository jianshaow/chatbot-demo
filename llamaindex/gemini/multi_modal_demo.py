import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

from common import google_mm_model as model_name

image_urls = [
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
    # "",
]

img_response = requests.get(image_urls[0])
print(image_urls[0])
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)
plt.show()

image_documents = load_image_urls(image_urls)
gemini_pro = GeminiMultiModal(model_name=model_name, transport="rest")
print("-" * 80)
print("multi-modal model:", model_name)

print("-" * 80)
# prompt = "Identify the city where this photo was taken."
prompt = "这张照片是在哪个城市拍摄的."
print("Question:", prompt)
complete_response = gemini_pro.complete(
    prompt=prompt,
    image_documents=image_documents,
)
print("Answer:", complete_response)

print("-" * 80)

# prompt = "Give me more context for this image"
prompt = "告诉我更多有关这张照片的内容"
print("Question:", prompt)
print("Answer:", end="")
stream_complete_response = gemini_pro.stream_complete(
    prompt=prompt,
    image_documents=image_documents,
)
for r in stream_complete_response:
    print(r.text, end="")
print("\n", "-" * 80, sep="")
