import asyncio

from llama_index.tools.mcp import McpToolSpec
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from common import sse_url


async def get_sse_tools_async():
    async with sse_client(url=sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await McpToolSpec(session).to_tool_list_async()
            return tools


def get_calc_sse_tools():
    return asyncio.run(get_sse_tools_async())


async def get_stdio_tools_async():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_tools/calc_stdio_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await McpToolSpec(session).to_tool_list_async()
            return tools


def get_calc_stdio_tools():
    return asyncio.run(get_stdio_tools_async())


if __name__ == "__main__":
    mcp_tools = get_calc_stdio_tools()
    for tool in mcp_tools:
        print(tool.metadata.name, ":", tool.metadata.description)
