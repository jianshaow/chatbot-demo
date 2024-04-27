SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

prompt_templates = {
    "vicuna": "{system_prompt} USER: {{prompt}}".format(system_prompt=SYSTEM_PROMPT),
    "llama": """[INST]<<SYS>>
{system_prompt}
<</SYS>

{{prompt}} [/INST]""".format(
        system_prompt=SYSTEM_PROMPT
    ),
}


def chat_prompt(prompt, model_type="vicuna"):
    return prompt_templates[model_type].format(prompt=prompt)


if __name__ == "__main__":
    import sys

    model_type = len(sys.argv) == 2 and sys.argv[1] or "vicuna"
    print(chat_prompt("who are you?", model_type=model_type))
