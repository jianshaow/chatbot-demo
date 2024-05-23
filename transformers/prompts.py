SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

prompt_templates = {
    "vicuna": "{system_prompt} USER: {user_prompt}",
    "llama": """[INST]<<SYS>>
{system_prompt}
<</SYS>

{user_prompt} [/INST]""",
}


def chat_prompt(user_prompt, system_prompt=SYSTEM_PROMPT, model_type="vicuna"):
    return prompt_templates[model_type].format(
        user_prompt=user_prompt, system_prompt=system_prompt
    )


if __name__ == "__main__":
    import sys

    model_type = len(sys.argv) == 2 and sys.argv[1] or "vicuna"
    print(chat_prompt("who are you?", model_type=model_type))
