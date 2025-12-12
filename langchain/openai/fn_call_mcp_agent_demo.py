import asyncio

from common import openai_fc_model as model_name
from common.models import demo_fn_call_agent_async
from common.openai import get_chat_model
from mcp_tools.calc_client import stdio_tools_context as tools_context


async def main():
    fn_call_model = get_chat_model(model=model_name)
    async with tools_context() as tools:
        await demo_fn_call_agent_async(fn_call_model, model_name, tools=tools)


asyncio.run(main())
