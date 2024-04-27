import os, torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM

model_name = os.environ.get("HF_MODEL_NAME", "lmsys/vicuna-7b-v1.5")

model_kwargs = {}
if os.environ.get("BNB_ENABLED", "false") == "true":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs.update({"quantization_config": quantization_config})

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    # generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name=model_name,
    model_name=model_name,
    model_kwargs=model_kwargs,
    system_prompt="You are a pirate with a colorful personality.",
    query_wrapper_prompt="USER: {query_str}",
)

print("-" * 80)
response = llm.complete("who are you?")
print(response.text)
print("\n" + "-" * 80)