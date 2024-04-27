SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

prompt_templates = {
    "vicuna": "{system_prompt}\nUSER: {{query_str}}".format(
        system_prompt=SYSTEM_PROMPT
    ),
    "llama": """[INST]<<SYS>>
{system_prompt}
<</SYS>

{{query_str}} [/INST]""".format(
        system_prompt=SYSTEM_PROMPT
    ),
}


def rag_template(model_type="vicuna"):
    return prompt_templates[model_type]


if __name__ == "__main__":
    import sys

    model_type = len(sys.argv) == 2 and sys.argv[1] or "vicuna"
    print(rag_template("who are you?", model_type=model_type))
