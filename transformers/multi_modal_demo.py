from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from common import hf_mm_model as model_name
from models import default_model_kwargs

print("-" * 80)
print("chat model:", model_name)

model_kwargs = default_model_kwargs()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", **model_kwargs
)

processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
            },
            {"type": "text", "text": "Identify the city where this photo was taken."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("-" * 80)
print(output_text[0])
