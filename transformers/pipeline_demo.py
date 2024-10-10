from transformers import pipeline
from common import hf_chat_model as model
from models import default_model_kwargs

pipe = pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    torch_dtype="auto",
    model_kwargs=default_model_kwargs(),
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate with a colorful personality.",
    },
    {"role": "user", "content": "what is your name?"},
]

response = pipe(messages, max_new_tokens=256)

print("-" * 80)
print(response[0]["generated_text"][-1]["content"])
print("-" * 80)
