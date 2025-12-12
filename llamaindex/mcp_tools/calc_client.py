import asyncio
from contextlib import asynccontextmanager

from llama_index.tools.mcp import McpToolSpec
from mcp import StdioServerParameters

from common import sse_url
from mcp_tools import sse_session, stdio_session


@asynccontextmanager
async def sse_tools_context():
    async with sse_session(sse_url) as session:
        tools = await McpToolSpec(session).to_tool_list_async()
        yield tools


def get_sse_tools():
    async def aget_sse_tools():
        async with sse_tools_context() as sse_tools:
            return sse_tools

    return asyncio.run(aget_sse_tools())


@asynccontextmanager
async def stdio_tools_context():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_tools/calc_stdio_server.py"],
    )
    async with stdio_session(server_params) as session:
        tools = await McpToolSpec(session).to_tool_list_async()
        yield tools


def get_stdio_tools():
    async def aget_stdio_tools():
        async with stdio_tools_context() as stdio_tools:
            return stdio_tools

    return asyncio.run(aget_stdio_tools())


if __name__ == "__main__":
    mcp_tools = get_stdio_tools()
    for tool in mcp_tools:
        print(tool.metadata.name, ":", tool.metadata.description)
