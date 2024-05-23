import os, sys, torch, prompts
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

model_name = os.environ.get("HF_MODEL", "lmsys/vicuna-7b-v1.5")
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = None
if os.environ.get("BNB_ENABLED", "false") == "true":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)

query_str = len(sys.argv) == 2 and sys.argv[1] or "Who are you?"
prompt = prompts.chat_prompt(query_str)

input_ids = tokenizer(
    prompt, return_tensors="pt", return_attention_mask=False
).input_ids.to(model.device)

generation_config = GenerationConfig.from_pretrained(model_name, max_length=1024)
tokens = model.generate(input_ids, generation_config=generation_config)

token_ids = tokens[0][input_ids.size(1) :]
response = tokenizer.decode(token_ids, skip_special_tokens=True)

print("-" * 80)
print(response)
print("-" * 80)

prompt = prompts.chat_prompt(
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
)

input_ids = tokenizer(
    prompt, return_tensors="pt", return_attention_mask=False
).input_ids.to(model.device)

tokens = model.generate(input_ids, generation_config=generation_config)

token_ids = tokens[0][input_ids.size(1) :]
response = tokenizer.decode(token_ids, skip_special_tokens=True)

print(response)
print("-" * 80)
