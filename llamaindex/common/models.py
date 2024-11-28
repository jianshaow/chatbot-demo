import os
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}
    bnb_enabled = os.environ.get("BNB_ENABLED", "false") == "true"
    if bnb_enabled:
        import torch
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return model_kwargs


def demo_chat(chat_model: LLM, model_name: str):
    print("-" * 80)
    print("chat model:", model_name)

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="What is your name"),
    ]

    print("-" * 80)
    response = chat_model.stream_chat(messages)
    for chunk in response:
        print(chunk.delta, end="")
    print("\n", "-" * 80, sep="")
