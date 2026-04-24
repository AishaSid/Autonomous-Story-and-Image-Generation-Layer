from __future__ import annotations

from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables before importing tools that read API keys.
load_dotenv()

from tools.character_tools import query_stock_footage
from tools.image_tools import generate_character_image
from tools.memory_tools import commit_memory
from tools.script_generation import generate_script_segment

mcp = FastMCP("writer-room-mcp-server")

mcp.tool()(generate_script_segment)
mcp.tool()(query_stock_footage)
mcp.tool()(generate_character_image)
mcp.tool()(commit_memory)


if __name__ == "__main__":
    mcp.run()
