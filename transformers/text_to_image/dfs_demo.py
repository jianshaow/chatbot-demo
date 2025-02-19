import torch
from diffusers import DiffusionPipeline

from common import hf_dfs_model as base_model
from common import hf_dfs_rf_model as refiner_model
from common.images import show_image

base = DiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    refiner_model,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

n_steps = 40
high_noise_frac = 0.8

prompt = "An astronaut riding a green horse"

image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images[0]
print(image.shape)
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
show_image(image)
# image.save("output/astronaut_rides_green_horse.png")
