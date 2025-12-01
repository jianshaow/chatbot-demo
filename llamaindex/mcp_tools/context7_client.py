import asyncio

from llama_index.core.tools.types import ToolOutput
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from mcp.types import CallToolResult, TextContent


async def get_http_tools_async():
    mcp_client = BasicMCPClient("https://mcp.context7.com/mcp")
    mcp_tool_spec = McpToolSpec(client=mcp_client)

    return await mcp_tool_spec.to_tool_list_async()


def get_http_tools():
    return asyncio.run(get_http_tools_async())


if __name__ == "__main__":
    mcp_tools = get_http_tools()
    for tool in mcp_tools:
        tool_name = tool.metadata.name
        print("tool name:", tool_name)
        print("-" * 80)
        print("tool description:")
        print(tool.metadata.description)
        if tool_name == "resolve-library-id":
            kwargs = {"libraryName": "spring-boot"}
            print("calling tool", tool_name, "with args:", kwargs)
            output: ToolOutput = tool.call(**kwargs)
            result: CallToolResult = output.raw_output
            text = "\n".join(
                [
                    block.text
                    for block in result.content
                    if isinstance(block, TextContent)
                ]
            )
            print("-" * 80)
            print("tool result:")
            print(text)
        print("=" * 80)
