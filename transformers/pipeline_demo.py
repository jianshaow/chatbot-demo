from transformers import pipeline, TextStreamer
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

streamer = TextStreamer(
    pipe.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("-" * 80)
pipe(messages, max_new_tokens=256, streamer=streamer)
print("-" * 80)
