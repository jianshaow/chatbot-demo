from common import hf_chat_model as model_name
from common.models import generate, new_model
from common.prompts import tokenizer_prompt

model, tokenizer = new_model(model_name)
print("-" * 80)
print("chat model:", model_name)

prompt = tokenizer_prompt(
    tokenizer,
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
)

response = generate(model, tokenizer, prompt, streaming=True)

print("-" * 80)
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
