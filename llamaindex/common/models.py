import os, sys
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

from common.fn_tools import tools
from common.functions import fns
from common.prompts import system_prompt, examples


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


def demo_embed(embed_model: BaseEmbedding, model_name: str):
    print("-" * 80)
    print("embed model:", model_name)

    question = (
        len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
    )
    embedding = embed_model.get_text_embedding(question)
    print("-" * 80)
    print("dimension:", len(embedding))
    print(embedding[:4])
    print("-" * 80)


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


def demo_fn_call(fn_call_model: FunctionCallingLLM, model_name: str):
    print("-" * 80)
    print("fn call model:", model_name)

    messages = [
        system_prompt,
        *examples,
        ChatMessage(role="user", content="What is (121 * 3) + 42?"),
    ]
    response = fn_call_model.chat_with_tools(tools, chat_history=messages)

    while response.message.additional_kwargs.get("tool_calls"):
        print("-" * 80)
        messages.append(response.message)
        for tool_call in response.message.additional_kwargs.get("tool_calls"):
            fn_name = tool_call["function"]["name"]
            fn = fns[fn_name]
            fn_args = tool_call["function"]["arguments"]
            print("=== Calling Function ===")
            print(
                "Calling function:",
                fn_name,
                "with args:",
                fn_args,
            )
            fn_result = fn(**fn_args)
            print("Got output:", fn_result)
            print("========================\n")
            tool_message = ChatMessage(
                content=str(fn_result),
                role="tool",
                additional_kwargs={"name": fn_name},
            )
            messages.append(tool_message)
        response = fn_call_model.chat_with_tools(tools, chat_history=messages)

    print("-" * 80)
    print(response.message.content)


def demo_recieve(
    embed_model: BaseEmbedding,
    model_name: str,
    data_path: str = "data",
    query="What did the author do growing up?",
):
    print("-" * 80)
    Settings.embed_model = embed_model
    print("-" * 80)
    print("embed model:", model_name)

    documents = SimpleDirectoryReader(data_path).load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True,
    )

    retriever = index.as_retriever(
        similarity_top_k=4,
    )
    question = len(sys.argv) == 2 and sys.argv[1] or query
    nodes = retriever.retrieve(question)
    for node in nodes:
        print("-" * 80)
        print(node)
