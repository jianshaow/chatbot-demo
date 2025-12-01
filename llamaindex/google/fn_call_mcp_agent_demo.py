import asyncio

from llama_index.llms.google_genai import GoogleGenAI

from common import google_chat_model as model
from common.models import demo_fn_call_agent_async
from mcp_tools.calc_client import stdio_tools_context as tools_context


async def main():
    fn_call_model = GoogleGenAI(model=model)
    async with tools_context() as tools:
        await demo_fn_call_agent_async(fn_call_model, model, tools=tools)


asyncio.run(main())
