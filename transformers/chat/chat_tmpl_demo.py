from common import hf_chat_model as model_name
from common.models import generate, new_model
from common.prompts import chat_prompt

model, tokenizer = new_model(model_name)
print("-" * 80)
print("chat model:", model_name)

prompt = chat_prompt(
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
    model_type="llama3",
)

response = generate(model, tokenizer, prompt, streaming=True)

print("-" * 80)
for chunk in response:
    print(chunk, end="", flush=True)
print("\n", "-" * 80, sep="")
