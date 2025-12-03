import asyncio
import json

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream, AgentWorkflow, ToolCallResult
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import BaseMemory
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import ImageNode
from llama_index.core.tools import RetrieverTool
from mcp.types import CallToolResult

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
from common.workflows import from_tools_or_functions


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


def demo_agent(
    embed_model: BaseEmbedding,
    embed_model_name: str,
    chat_model: LLM,
    chat_model_name: str,
    data_path: str = "data/default",
    query="What did the author do growing up?",
):
    print("-" * 80)
    print("embed model:", embed_model_name)
    print("chat model:", chat_model_name)
    print("-" * 80)

    index = __get_index(embed_model, data_path)
    retriever_tool = RetrieverTool.from_defaults(index.as_retriever())
    agent = from_tools_or_functions([retriever_tool], chat_model)
    __run_agent(agent, query)


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

    has_tool_call = True
    while has_tool_call:
        print("-" * 80)
        messages.append(response.message)
        has_tool_call = False
        for block in response.message.blocks:
            if isinstance(block, ToolCallBlock):
                has_tool_call = True

                tool_call_id = block.tool_call_id
                fn_name = block.tool_name
                fn_args = (
                    json.loads(block.tool_kwargs)
                    if isinstance(block.tool_kwargs, str)
                    else block.tool_kwargs
                )
                print("=== Calling Function ===")
                print(
                    "Calling function:",
                    fn_name,
                    "with args:",
                    fn_args,
                )
                fn = fns[fn_name]
                fn_args = (
                    json.loads(block.tool_kwargs)
                    if isinstance(block.tool_kwargs, str)
                    else block.tool_kwargs
                )
                fn_result = fn(**fn_args)
                print("Got output:", fn_result)
                print("========================\n")
                tool_message = ChatMessage(
                    content=str(fn_result),
                    role="tool",
                    additional_kwargs={
                        "name": fn_name,
                        "tool_call_id": tool_call_id,
                    },
                )
                messages.append(tool_message)
            elif isinstance(block, TextBlock):
                print(block.text)
        if has_tool_call:
            response = fn_call_model.chat_with_tools(
                tools, chat_history=messages, allow_parallel_tool_calls=True
            )


def demo_fn_call_agent(
    fn_call_model: LLM, model: str, tools=None, query=tool_call_question
):
    print("-" * 80)
    print("fn call model:", model)
    print("-" * 80)

    if tools is None:
        tools = calc_tools

    agent = from_tools_or_functions(tools, fn_call_model)
    __run_agent(agent, query)


async def demo_fn_call_agent_async(
    fn_call_model: LLM, model: str, tools=None, query=tool_call_question
):
    print("-" * 80)
    print("fn call model:", model)
    print("-" * 80)

    if tools is None:
        tools = calc_tools

    agent = from_tools_or_functions(tools, fn_call_model)
    await __run_agent_async(agent, query)


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

    index = __get_index(embed_model, data_path)

    retriever = index.as_retriever(
        similarity_top_k=2,
    )
    question = get_args(1, query)
    nodes = retriever.retrieve(question)
    for node in nodes:
        print("-" * 80)
        print(node)


def __get_index(embed_model: BaseEmbedding, data_path: str) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(data_path).load_data(show_progress=True)
    for doc in documents:
        doc.excluded_embed_metadata_keys.append("file_path")
        doc.excluded_llm_metadata_keys.append("file_path")
    return VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True,
    )


async def __run_agent_async(
    agent: AgentWorkflow, question: str, memory: BaseMemory | None = None
):
    handler = agent.run(question, memory=memory)
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)
            print("Arguments to the tool: ", event.tool_kwargs)
            output = event.tool_output.raw_output
            if isinstance(output, list):
                print("Tool output size: ", len(event.tool_output.raw_output))
            elif isinstance(output, CallToolResult):
                print("Tool output size: ", len(output.content))
            print("-" * 80)
    print("\n", "-" * 80, sep="")


def __run_agent(
    agent: AgentWorkflow,
    question: str,
    memory: BaseMemory | None = None,
):

    asyncio.run(__run_agent_async(agent, question, memory))
