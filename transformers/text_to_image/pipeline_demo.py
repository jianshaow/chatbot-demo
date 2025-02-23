import torch
from diffusers import AutoPipelineForText2Image

from common import hf_t2i_model as model
from common.images import show_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

show_image(image)
