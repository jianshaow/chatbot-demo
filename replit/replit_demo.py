import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(
    "replit/replit-code-v1_5-3b", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "replit/replit-code-v1_5-3b",
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
)

x = tokenizer.encode("def fibonacci(n): ", return_tensors="pt").to("cuda:0")
y = model.generate(
    x,
    max_length=100,
    do_sample=True,
    top_p=0.95,
    top_k=4,
    temperature=0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

# decoding
generated_code = tokenizer.decode(
    y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("======================================================")
print(generated_code)
