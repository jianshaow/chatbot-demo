import textwrap
from typing import Any

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from common import get_args, get_env_bool
from common.calc_func import fns
from common.prompts import (
    CHAT_SYSTEM,
    FN_CALL_SYSTEM,
    chat_question,
    embed_question,
    examples,
)
from common.prompts import fn_adv_question_message as question_message
from common.prompts import fn_call_adv_question as fn_call_question
from common.prompts import (
    fn_call_system_message,
    mm_image_url,
    mm_question1,
    mm_question2,
)
from common.tools import calc_tools


def default_model_kwargs() -> dict[str, str]:
    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if get_env_bool("BNB_ENABLED"):
        import torch
        from transformers.utils.quantization_config import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return model_kwargs


def demo_embed(embed_model: Embeddings, model: str, query=embed_question):
    print("-" * 80)
    print("embed model:", model)

    question = get_args(1, query)
    embedding = embed_model.embed_query(question)
    print("-" * 80)
    print("dimension:", len(embedding))
    print(embedding[:4])
    print("-" * 80, sep="")


def demo_chat(chat_model: BaseChatModel, model: str):
    print("-" * 80)
    print("chat model:", model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHAT_SYSTEM),
            ("user", "{input}"),
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | chat_model | output_parser

    print("-" * 80)
    response = chain.stream({"input": chat_question})
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n", "-" * 80, sep="")


def demo_fn_call(
    fn_call_model: BaseChatModel, model: str, tools=None, with_few_shot=False
):
    print("-" * 80)
    print("fn call model:", model)

    if tools is None:
        tools = calc_tools

    llm_with_tools = fn_call_model.bind_tools(tools)

    messages = []
    if with_few_shot:
        messages.append(fn_call_system_message)
        messages.extend(examples)
    messages.append(question_message)

    response = llm_with_tools.invoke(messages)

    while response.tool_calls:
        print("-" * 80)
        messages.append(response)

        for tool_call in response.tool_calls:
            print("=== Calling Function ===")
            print(
                "Calling function:",
                tool_call["name"],
                "with args:",
                tool_call["args"],
            )
            fn = fns[tool_call["name"]]
            fn_result = fn.invoke(tool_call)
            print("Got output:", fn_result.content)
            print("========================\n")
            messages.append(fn_result)
        response = llm_with_tools.invoke(messages)

    print("-" * 80)
    print(response.content)


def demo_fn_call_agent(fn_call_model: BaseChatModel, model: str, with_few_shot=False):
    print("-" * 80)
    print("fn call model:", model)

    agent = create_agent(fn_call_model, calc_tools, system_prompt=FN_CALL_SYSTEM)

    messages = []
    if with_few_shot:
        messages.extend(examples)
    messages.append({"role": "user", "content": fn_call_question})

    response = agent.invoke({"messages": messages})
    print("-" * 80)
    print(response["messages"][-1].content)


def demo_multi_modal(mm_model: BaseChatModel, model: str, image_data=None):
    print("-" * 80)
    print("multi-modal model:", model)

    if image_data:
        image_placeholder = "data:image/jpeg;base64,{image_data}"
        query_args = {"image_data": image_data}
    else:
        image_placeholder = "{image_url}"
        query_args = {"image_url": mm_image_url}

    template = HumanMessagePromptTemplate.from_template(
        [{"text": "{input}"}, {"image_url": {"url": image_placeholder}}]
    )
    prompt = ChatPromptTemplate.from_messages([template])

    output_parser = StrOutputParser()
    chain = prompt | mm_model | output_parser

    print("-" * 80)
    print("Question:", mm_question1)
    query_args.update({"input": mm_question1})
    response = chain.invoke(query_args)
    print("Answer:", response)
    print("-" * 80)

    print("Question:", mm_question2)
    print("Answer:", end="")
    query_args.update({"input": mm_question2})
    response = chain.stream(query_args)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n", "-" * 80, sep="")


def demo_retrieve(
    embed_model: Embeddings,
    model: str,
    data_path: str = "data/default",
    query=embed_question,
):
    loader = DirectoryLoader(data_path)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    documents = text_splitter.split_documents(loader.load())
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
    )
    print("-" * 80)
    print("embed model:", model)

    question = get_args(1, query)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    for doc in docs:
        print("-" * 80)
        print(textwrap.fill(doc.page_content[:347] + "..."))
    print("-" * 80)
