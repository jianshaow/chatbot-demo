import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = None
if os.environ.get("BNB_ENABLED", "false") == "true":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    device_map="auto",
)

prompt = """[INST]<<SYS>>
You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
<</SYS>

who are you?[/INST]
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs)
completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
response = tokenizer.decode(completion_tokens, skip_special_tokens=True)

print(response)
