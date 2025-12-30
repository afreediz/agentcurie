from controller import ToolsController
from mcp_client.service import MCPClient
import asyncio


async def main() -> None:
    # Connect to an MCP server
    controller = ToolsController()
    mcp_client = MCPClient(
        server_name="my-server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem",r"C:\path\to\your\folder"]
    )

    # Register all MCP tools as browser-use actions
    await mcp_client.register_to_controller(controller)
    print(controller.registry.get_prompt_description())

if __name__ == "__main__":
    asyncio.run(main())