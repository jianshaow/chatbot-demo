from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.agent import AgentRunner
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

from common.fn_tools import tools
from common.functions import fns
from common.prompts import system_message, examples, question_message
from common import demo_image_url as image_url, get_args, get_env_bool


def default_model_kwargs() -> dict[str, str]:
    model_kwargs = {}
    bnb_enabled = get_env_bool("BNB_ENABLED")
    if bnb_enabled:
        import torch
        from transformers import BitsAndBytesConfig

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


def demo_fn_call(
    fn_call_model: FunctionCallingLLM, model_name: str, with_few_shot=False
):
    print("-" * 80)
    print("fn call model:", model_name)

    messages = []
    if with_few_shot:
        messages.append(system_message)
        messages.extend(examples)
    messages.append(question_message)

    response = fn_call_model.chat_with_tools(tools, chat_history=messages)

    while response.message.additional_kwargs.get("tool_calls"):
        print("-" * 80)
        messages.append(response.message)
        for tool_call in response.message.additional_kwargs.get("tool_calls"):
            id, fn_name, fn_args = __get_tool_call_info(tool_call)
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
                additional_kwargs={"name": fn_name, "tool_call_id": id},
            )
            messages.append(tool_message)
        response = fn_call_model.chat_with_tools(tools, chat_history=messages)

    print("-" * 80)
    print(response.message.content)


def __get_tool_call_info(tool_call):
    id = getattr(tool_call, "id", None)
    if hasattr(tool_call, "function"):
        import json

        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
    else:
        fn_name = tool_call["function"]["name"]
        fn_args = tool_call["function"]["arguments"]

    return id, fn_name, fn_args


def demo_fn_call_agent(fn_call_model: LLM, model_name: str, with_few_shot=False):
    print("-" * 80)
    print("fn call model:", model_name)

    messages: list[ChatMessage] = []
    if with_few_shot:
        messages.append(system_message)
        messages.extend(examples)
    messages.append(question_message)

    agent = AgentRunner.from_llm(tools, fn_call_model, verbose=True)
    response = agent.chat(message=messages[-1].content, chat_history=messages[:-1])

    print("-" * 80)
    print(response)


def demo_multi_modal(mm_model: MultiModalLLM, model_name: str):
    print("-" * 80)
    print("multi-modal model:", model_name)

    image_documents = load_image_urls([image_url])
    print("-" * 80)

    prompt = "Identify the city where this photo was taken."
    # prompt = "这张照片是在哪个城市拍摄的."
    print("Question:", prompt)
    complete_response = mm_model.complete(
        prompt=prompt,
        image_documents=image_documents,
    )
    print("Answer:", complete_response)

    print("-" * 80)

    prompt = "Give me more context for this image"
    # prompt = "给我更多这张照片的上下文"
    print("Question:", prompt)
    print("Answer:", end="")
    stream_complete_response = mm_model.stream_complete(
        prompt=prompt,
        image_documents=image_documents,
    )
    for r in stream_complete_response:
        print(r.delta, end="")
    print("\n", "-" * 80, sep="")


def demo_recieve(
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
