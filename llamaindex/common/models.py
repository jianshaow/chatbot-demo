from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import AgentRunner
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import ImageNode

from common import demo_image_url as image_url
from common import get_args, get_env_bool
from common.calc_func import fns
from common.prompts import (
    examples,
    question_message,
    system_message,
    tool_call_question,
)
from common.tools import calc_tools


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}
    bnb_enabled = get_env_bool("BNB_ENABLED")
    if bnb_enabled:
        import torch
        from transformers.utils.quantization_config import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return model_kwargs


def demo_embed(
    embed_model: BaseEmbedding,
    model_name: str,
    query="What did the author do growing up?",
):
    print("-" * 80)
    print("embed model:", model_name)

    question = get_args(1, query)
    embedding = embed_model.get_text_embedding(question)
    print("-" * 80)
    print("dimension:", len(embedding))
    print(embedding[:4])
    print("-" * 80)


def demo_chat(chat_model: LLM, model: str):
    print("-" * 80)
    print("chat model:", model)

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


def demo_fn_call(
    fn_call_model: FunctionCallingLLM, model_name: str, tools=None, with_few_shot=False
):
    print("-" * 80)
    print("fn call model:", model_name)

    if tools is None:
        tools = calc_tools

    messages = []
    if with_few_shot:
        messages.append(system_message)
        messages.extend(examples)
    messages.append(question_message)

    response = fn_call_model.chat_with_tools(
        tools, chat_history=messages, allow_parallel_tool_calls=True
    )

    while response.message.additional_kwargs.get("tool_calls"):
        print("-" * 80)
        messages.append(response.message)
        if tool_calls := response.message.additional_kwargs.get("tool_calls"):
            for tool_call in tool_calls:
                tool_call_id, fn_name, fn_args = __get_tool_call_info(tool_call)
                print("=== Calling Function ===")
                print(
                    "Calling function:",
                    fn_name,
                    "with args:",
                    fn_args,
                )
                fn = fns[fn_name]
                fn_result = fn(**fn_args)
                print("Got output:", fn_result)
                print("========================\n")
                tool_message = ChatMessage(
                    content=str(fn_result),
                    role="tool",
                    additional_kwargs={"name": fn_name, "tool_call_id": tool_call_id},
                )
                messages.append(tool_message)
        response = fn_call_model.chat_with_tools(
            calc_tools, chat_history=messages, allow_parallel_tool_calls=True
        )

    print("-" * 80)
    print(response.message.content)


def __get_tool_call_info(tool_call):
    tool_call_id = getattr(tool_call, "id", None)
    if hasattr(tool_call, "function"):
        import json

        fn_name = tool_call.function.name
        fn_args = (
            json.loads(tool_call.function.arguments)
            if isinstance(tool_call.function.arguments, str)
            else tool_call.function.arguments
        )
    else:
        fn_name = str(tool_call.name)
        fn_args = dict(tool_call.args)

    return tool_call_id, fn_name, fn_args


def demo_fn_call_agent(fn_call_model: LLM, model: str):
    print("-" * 80)
    print("fn call model:", model)

    agent = AgentRunner.from_llm(calc_tools, fn_call_model, verbose=True)
    response = agent.chat(message=tool_call_question)

    print("-" * 80)
    print(response)


def demo_multi_modal_legacy(mm_model: MultiModalLLM, model: str, streaming=True):
    print("-" * 80)
    print("multi-modal model:", model)
    print("-" * 80)

    image_documents = [ImageNode(image_path=image_url)]

    prompt = "Identify the city where this photo was taken."
    # prompt = "这张照片是在哪个城市拍摄的."
    print("Question:", prompt)
    complete_response = mm_model.complete(
        prompt=prompt,
        image_documents=image_documents,  # type: ignore
    )
    print("Answer:", complete_response)
    print("-" * 80)

    if not streaming:
        return

    prompt = "Give me more context for this image"
    # prompt = "给我更多这张照片的上下文"
    print("Question:", prompt)
    print("Answer:", end="")
    stream_complete_response = mm_model.stream_complete(
        prompt=prompt,
        image_documents=image_documents,  # type: ignore
    )
    for r in stream_complete_response:
        print(r.delta, end="")
    print("\n", "-" * 80, sep="")


def demo_multi_modal(mm_model: LLM, model: str, image_block=None, streaming=True):
    print("-" * 80)
    print("multi-modal model:", model)
    print("-" * 80)

    image_block = image_block if image_block else ImageBlock(url=image_url)

    prompt = "Identify the city where this photo was taken."
    # prompt = "这张照片是在哪个城市拍摄的."
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                image_block,
                TextBlock(text=prompt),
            ],
        )
    ]
    print("Question:", prompt)
    complete_response = mm_model.chat(messages)
    print("Answer:", complete_response)
    print("-" * 80)

    if not streaming:
        return

    prompt = "Give me more context for this image"
    # prompt = "给我更多这张照片的上下文"
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                image_block,
                TextBlock(text=prompt),
            ],
        )
    ]
    print("Question:", prompt)
    print("Answer:", end="")
    stream_complete_response = mm_model.stream_chat(messages)
    for r in stream_complete_response:
        print(r.delta, end="")
    print("\n", "-" * 80, sep="")


def demo_retrieve(
    embed_model: BaseEmbedding,
    model_name: str,
    data_path: str = "data/default",
    query="What did the author do growing up?",
):
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
    question = get_args(1, query)
    nodes = retriever.retrieve(question)
    for node in nodes:
        print("-" * 80)
        print(node)
