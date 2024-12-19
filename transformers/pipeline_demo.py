from transformers import pipeline, TextStreamer
from common import hf_chat_model as model
from models import default_model_kwargs

generate = pipeline(
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
    generate.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("-" * 80)
generate(
    messages,
    max_new_tokens=256,
    streamer=streamer,
    pad_token_id=generate.tokenizer.eos_token_id,
)
print("-" * 80)
