import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp_tools/stdio_server.py"],
)


async def get_tools_async():
    async with stdio_client(server_params) as (read, write):
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
