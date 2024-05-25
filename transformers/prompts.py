from transformers import AutoTokenizer, PreTrainedTokenizer

SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

prompt_templates = {
    "vicuna": "{system_prompt} USER: {user_prompt}",
    "llama": """<s> [INST] <<SYS>>
{system_prompt}
<</SYS>

{user_prompt} [/INST]""",
}


def chat_prompt(user_prompt, system_prompt=SYSTEM_PROMPT, model_type="vicuna"):
    return prompt_templates[model_type].format(
        user_prompt=user_prompt, system_prompt=system_prompt
    )


def tokenizer_prompt(
    tokenizer: PreTrainedTokenizer, user_prompt: str, system_prompt=SYSTEM_PROMPT
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tokens = tokenizer.apply_chat_template(messages)
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    import sys, os

    model_type = len(sys.argv) == 2 and sys.argv[1] or "vicuna"
    print("-" * 80)
    print(chat_prompt("who are you?", model_type=model_type))

    model_name = os.environ.get("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("-" * 80)
    print(tokenizer_prompt(tokenizer, "who are you?"))
    print("-" * 80)
