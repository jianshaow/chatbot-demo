import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_http_tools_async():
    client = MultiServerMCPClient(
        {
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
            }
        }
    )
    return await client.get_tools()


def get_http_tools():
    return asyncio.run(get_http_tools_async())


async def main():
    mcp_tools = await get_http_tools_async()
    for tool in mcp_tools:
        tool_name = tool.name
        print("tool name:", tool_name)
        print("-" * 80)
        print("tool description:")
        print(tool.description)
        if tool_name == "resolve-library-id":
            kwargs = {"libraryName": "spring-boot"}
            print("calling tool", tool_name, "with args:", kwargs)
            output = await tool.ainvoke(kwargs)
            text = "\n".join([block["text"] for block in output])
            print("-" * 80)
            print("tool result:")
            print(text)
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
