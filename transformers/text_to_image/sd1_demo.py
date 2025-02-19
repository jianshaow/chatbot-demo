import torch
from diffusers import StableDiffusionPipeline

from common import hf_sd1_model as model
from common.images import show_image

pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

show_image(image)
# image.save("output/astronaut_rides_horse.png")
