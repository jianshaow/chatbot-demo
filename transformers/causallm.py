import os, sys, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

model_name = os.environ.get("HF_MODEL_NAME", "lmsys/vicuna-7b-v1.5")
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

prompt = f"""[INST]<<SYS>>
You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
<</SYS>

{query_str}[/INST]
"""

input_ids = tokenizer(
    prompt, return_tensors="pt", return_attention_mask=False
).input_ids.to(model.device)

generation_config = GenerationConfig.from_pretrained(model_name, max_length=1024)
tokens = model.generate(input_ids, generation_config=generation_config)

token_ids = tokens[0][input_ids.size(1) :]
response = tokenizer.decode(token_ids, skip_special_tokens=True)

print(response)
