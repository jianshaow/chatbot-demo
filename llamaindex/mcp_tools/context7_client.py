import asyncio

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def get_sse_tools_async():
    mcp_client = BasicMCPClient("https://mcp.context7.com/mcp")
    mcp_tool_spec = McpToolSpec(client=mcp_client)

    return await mcp_tool_spec.to_tool_list_async()


def get_sse_tools():
    return asyncio.run(get_sse_tools_async())


if __name__ == "__main__":
    mcp_tools = get_sse_tools()
    for tool in mcp_tools:
        print("tool name:", tool.metadata.name)
        print("-" * 80)
        print(tool.metadata.description)
        print("=" * 80)
