prompt_templates = {
    "default": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt}",
    "llama": """[INST]<<SYS>>
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<</SYS>

{prompt} [/INST]""",
}


def chat_prompt(prompt, model_type="default"):
    return prompt_templates[model_type].format(prompt=prompt)


if __name__ == "__main__":
    print(chat_prompt("who are you?"))
