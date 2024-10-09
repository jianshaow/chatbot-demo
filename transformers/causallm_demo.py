import common, prompts

model_name = common.hf_chat_model
tokenizer = common.new_tokenizer(model_name)
model = common.new_model(model_name)

print("-" * 80)
prompt = prompts.tokenizer_prompt(
    tokenizer,
    system_prompt="You are a pirate with a colorful personality.",
    user_prompt="what is your name?",
)

response = common.generate(model, tokenizer, prompt)

print(response)
print("-" * 80)
