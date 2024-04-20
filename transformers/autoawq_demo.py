import os, sys, prompts
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = os.environ.get("HF_MODEL_NAME", "TheBloke/vicuna-13B-v1.5-AWQ")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAWQForCausalLM.from_quantized(
    model_name, fuse_layers=True, safetensors=True, device_map="auto"
)

query_str = len(sys.argv) == 2 and sys.argv[1] or "Who are you?"
prompt = prompts.chat_prompt(query_str)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

tokens = model.generate(input_ids)
token_ids = tokens[0][input_ids.size(1) :]

response = tokenizer.decode(token_ids, skip_special_tokens=True)

print(response)
