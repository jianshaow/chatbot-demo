import asyncio

import chromadb
import rag_config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream, AgentWorkflow, ToolCallResult
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import RetrieverTool
from llama_index.vector_stores.chroma import ChromaVectorStore

config = rag_config.get_config()

Settings.embed_model = config.embed_model()
Settings.llm = config.chat_model()
print("-" * 80)
print("embed model:", config.embed_model_name)
print("chat model:", config.chat_model_name)
print("-" * 80)

client = chromadb.PersistentClient(path=config.vector_db_path)
chroma_collection = client.get_or_create_collection(config.vector_db_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

retriever_tool = RetrieverTool.from_defaults(index.as_retriever())
agent = AgentWorkflow.from_tools_or_functions([retriever_tool], Settings.llm)
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)


async def __run_agent(user_msg):
    handler = agent.run(user_msg, memory=memory)
    stream_started = False
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            if not stream_started and event.delta != "":
                print("-" * 80)
                print("Answer: ", end="")
                stream_started = True
            print(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallResult):
            if stream_started:
                print()
                stream_started = False
            print("-" * 80)
            print("Tool called: ", event.tool_name)
            print("Arguments: ", event.tool_kwargs)
            nodes = event.tool_output.raw_output
            for node in nodes:
                print("=" * 80)
                print(node)
        else:
            if stream_started:
                print()
                stream_started = False

    print("-" * 80)


async def __main():
    while True:
        user_input = input("User: ")
        if user_input == "bye":
            break
        await __run_agent(user_input)


if __name__ == "__main__":
    asyncio.run(__main())
