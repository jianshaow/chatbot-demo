import os, sys
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from common import demo_image_url as image_url
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


def demo_fn_call_agent(fn_call_model: BaseChatModel, model_name: str):
    print("-" * 80)
    print("fn call model:", model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(fn_call_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": "What is (121 * 3) + 42?"})
    print("-" * 80)
    print(response["output"])


def demo_multi_modal(mm_model: BaseChatModel, model_name: str):
    print("-" * 80)
    print("multi-modal model:", model_name)

    template = HumanMessagePromptTemplate.from_template(
        [{"text": "{input}"}, {"image_url": {"url": "{image_url}"}}]
    )
    prompt = ChatPromptTemplate.from_messages([template])

    output_parser = StrOutputParser()
    chain = prompt | mm_model | output_parser

    print("-" * 80)
    input = "Identify the city where this photo was taken."
    print("Question:", input)
    response = chain.invoke({"input": input, "image_url": image_url})
    print("Answer:", response)
    print("-" * 80)

    input = "Give me more context for this image."
    print("Question:", input)
    print("Answer:", end="")
    response = chain.stream({"input": input, "image_url": image_url})
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n", "-" * 80, sep="")


def demo_recieve(embed_model: Embeddings, model_name: str):
    loader = DirectoryLoader("data")
    data = loader.load()

    text_splitter = CharacterTextSplitter()
    documents = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
    )
    print("-" * 80)
    print("embed model:", model_name)

    question = (
        len(sys.argv) == 2 and sys.argv[1] or "What did the author do growing up?"
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    for doc in docs:
        print("-" * 80)
        print(doc.page_content[:80])
