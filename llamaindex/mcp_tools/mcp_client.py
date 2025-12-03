import asyncio
from typing import List, Optional

from llama_index.tools.mcp import BasicMCPClient
from mcp.types import TextContent


def get_docker_mcp_client(
    image: str,
    extra_docker_args: List[str] | None = None,
    command: str | None = None,
    args: List[str] | None = None,
):
    mcp_command = "docker"
    mcp_args = ["run", "--rm", "-i"]
    if extra_docker_args:
        mcp_args.extend(extra_docker_args)
    mcp_args.append(image)
    if command:
        mcp_args.append(command)
    if args:
        mcp_args.extend(args)
    return BasicMCPClient(mcp_command, args=mcp_args)


async def call_tool(
    client: BasicMCPClient,
    tool_name: str,
    arguments: Optional[dict] = None,
):
    result = await client.call_tool(tool_name, arguments=arguments)
    for content in result.content:
        if isinstance(content, TextContent):
            print(content.text)


if __name__ == "__main__":
    mcp_client = get_docker_mcp_client(
        "mcp/filesystem",
        [
            "-v",
            "/:/workspace",
        ],
        args=["/workspace"],
    )
    asyncio.run(
        call_tool(mcp_client, "list_directory", arguments={"path": "/workspace"})
    )
