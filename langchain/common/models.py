import os, sys
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from common.fn_tools import tools
from common.functions import fns


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


def demo_embed(embed_model: Embeddings, model_name: str):
    print("-" * 80)
    print("embed model:", model_name)

    question = (
        len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
    )
    embedding = embed_model.embed_query(question)
    print("-" * 80)
    print("dimension:", len(embedding))
    print(embedding[:4])
    print("-" * 80, sep="")


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


def demo_fn_call(fn_call_model: BaseChatModel, model_name: str):
    print("-" * 80)
    print("fn call model:", model_name)

    llm_with_tools = fn_call_model.bind_tools(tools)

    query = "What is (121 * 3) + 42?"
    messages = [HumanMessage(query)]
    response = llm_with_tools.invoke(messages)

    while response.tool_calls:
        print("-" * 80)
        messages.append(response)

        for tool_call in response.tool_calls:
            fn = fns[tool_call["name"]]
            print("=== Calling Function ===")
            print(
                "Calling function:",
                tool_call["name"],
                "with args:",
                tool_call["args"],
            )
            fn_result = fn.invoke(tool_call)
            print("Got output:", fn_result.content)
            print("========================\n")
            messages.append(fn_result)
        response = llm_with_tools.invoke(messages)

    print("-" * 80)
    print(response.content)
