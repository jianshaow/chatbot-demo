import os, sys, common, prompts

model_name = os.environ.get("HF_AWQ_MODEL", "TheBloke/vicuna-13B-v1.5-AWQ")

tokenizer = common.new_tokenizer(model_name)
model = common.new_model(model_name, bnb_enabled=False)

query_str = len(sys.argv) == 2 and sys.argv[1] or "Who are you?"
prompt = prompts.chat_prompt(query_str)

response = common.generate(model, tokenizer, prompt)

print("-" * 80)
print(response)

prompt = prompts.chat_prompt(
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
)

response = common.generate(model, tokenizer, prompt)

print("-" * 80)
print(response)
print("-" * 80)
