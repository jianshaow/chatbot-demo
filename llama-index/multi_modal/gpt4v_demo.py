import requests, os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

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
model_name = os.environ.get("OPENAI_MODEL", "gpt-4-vision-preview")
mm_model = OpenAIMultiModal(model=model_name)

print("-" * 80)
prompt = "Identify the city where this photo was taken."
# prompt = "这张照片是在哪个城市拍摄的."
print("Question:", prompt)
complete_response = mm_model.complete(
    prompt=prompt,
    image_documents=image_documents,
)
print("Answer:", complete_response)

print("-" * 80)

prompt = "Give me more context for this image"
# prompt = "给我更多这张照片的上下文"
print("Question:", prompt)
print("Answer:", end="")
stream_complete_response = mm_model.stream_complete(
    prompt=prompt,
    image_documents=image_documents,
)
for r in stream_complete_response:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
