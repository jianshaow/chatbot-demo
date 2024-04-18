from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
)

prompt = "hello"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs)
response = tokenizer.decode(outputs[0])

print(response)
