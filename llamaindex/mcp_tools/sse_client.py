import asyncio

from llama_index.tools.mcp import McpToolSpec
from mcp import ClientSession
from mcp.client.sse import sse_client

from common import sse_url


async def get_tools_async():
    async with sse_client(url=sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await McpToolSpec(session).to_tool_list_async()
            return tools


def get_tools():
    return asyncio.run(get_tools_async())


if __name__ == "__main__":
    mcp_tools = get_tools()
    for tool in mcp_tools:
        print(tool.metadata.name, ":", tool.metadata.description)
