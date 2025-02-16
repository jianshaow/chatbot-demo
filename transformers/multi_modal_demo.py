from common import hf_mm_model as model_name
from models import image_text_to_text, new_multi_modal_model
from prompts import image_text_prompt

print("-" * 80)
print("chat model:", model_name)

model, processor, config = new_multi_modal_model(model_name)

image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"
qestion = "Identify the city where this photo was taken."
images, text = image_text_prompt(image_url, qestion, processor, config)
response = image_text_to_text(model, processor, images, text)
print("-" * 80)
print(response)
