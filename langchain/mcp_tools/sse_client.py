import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client


async def get_tools_async():
    async with sse_client(url="http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools


def get_tools():
    return asyncio.run(get_tools_async())


if __name__ == "__main__":
    mcp_tools = get_tools()
    for tool in mcp_tools:
        print(tool.name, ":", tool.description)
