import sys, common, prompts

model_name = common.hf_chat_model
tokenizer = common.new_tokenizer(model_name)
model = common.new_model(model_name)

query_str = len(sys.argv) == 2 and sys.argv[1] or "Who are you?"
prompt = prompts.tokenizer_prompt(tokenizer, query_str)

response = common.generate(model, tokenizer, prompt)

print("-" * 80)
print(response)

prompt = prompts.tokenizer_prompt(
    tokenizer,
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
)

response = common.generate(model, tokenizer, prompt)

print("-" * 80)
print(response)
print("-" * 80)
