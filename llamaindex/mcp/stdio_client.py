import asyncio

from llama_index.tools.mcp import McpToolSpec

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp/stdio_server.py"],
)


async def get_tools_async():
    async with stdio_client(server_params) as (read, write):
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
