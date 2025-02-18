import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline

from common import hf_dfs_model as model

pipe = DiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe.to("cuda")

prompt = "An astronaut riding a green horse"

image = pipe(prompt=prompt).images[0]
plt.imshow(image)
plt.show()
