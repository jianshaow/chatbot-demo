import os, torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM

from common import hf_chat_model as model_name

model_kwargs = {}
if os.getenv("BNB_ENABLED", "false") == "true":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs["quantization_config"] = quantization_config

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    tokenizer_name=model_name,
    model_name=model_name,
    model_kwargs=model_kwargs,
    system_prompt="You are a pirate with a colorful personality.",
)
print("-" * 80)
print("chat model:", model_name)

resp = llm.stream_complete("What is your name?")
print("-" * 80)
for r in resp:
    print(r.delta, end="")
print("\n", "-" * 80, sep="")
