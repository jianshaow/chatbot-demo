from io import BytesIO

import requests
from PIL import Image

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

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
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    user_prompt: str,
    system_prompt=SYSTEM_PROMPT,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def image_text_prompt(image_url: str, text: str, processor, config):
    architecture = config.architectures[0]
    if "Phi3VForCausalLM" in architecture:
        images, text = phi3v_prompt(image_url, text, processor)
    # elif "Florence2ForConditionalGeneration" in architecture:
    #     pass
    elif (
        "Qwen2_5_VLForConditionalGeneration" in architecture
        or "Qwen2VLForConditionalGeneration" in architecture
    ):
        images, text = qwen2vl_prompt(image_url, text, processor)
    # elif "PaliGemmaForConditionalGeneration" in architecture:
    #     pass
    # elif "MllamaForConditionalGeneration" in architecture:
    #     pass
    else:
        raise ValueError(f"architecture {architecture} not supported")
    return images, text


def phi3v_prompt(image_url: str, text: str, processor):
    image = Image.open(BytesIO(requests.get(image_url, timeout=10).content))
    messages = [{"role": "user", "content": f"<|image_1|>\n{text}"}]
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return [image], text


def qwen2vl_prompt(image_url: str, text: str, processor):
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    images, *_ = process_vision_info(messages)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return images, text


def main():
    import sys

    from common import hf_chat_model as model_name

    if len(sys.argv) == 2:
        model_type = sys.argv[1] or "vicuna"
        print("-" * 80)
        print(chat_prompt("who are you?", model_type=model_type))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("-" * 80)
        print(tokenizer_prompt(tokenizer, "who are you?"))
        print("-" * 80)


if __name__ == "__main__":
    main()
