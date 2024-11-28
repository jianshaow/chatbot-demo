import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel


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


def demo_chat(chat_model: BaseChatModel, model_name: str):
    print("-" * 80)
    print("chat model:", model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a pirate with a colorful personality."),
            ("user", "{input}"),
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | chat_model | output_parser

    print("-" * 80)
    response = chain.stream({"input": "What is your name?"})
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n", "-" * 80, sep="")
