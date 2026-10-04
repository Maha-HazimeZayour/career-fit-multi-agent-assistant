"""MCP client: lists the server's tools and calls them over stdio (Week 4, Lesson 3)."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_job_server.py"


async def _run(action) -> Any:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=dict(os.environ),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


def list_tools_via_mcp() -> list[dict[str, Any]]:
    """Ask the MCP server which tools it offers."""
    result = asyncio.run(_run(lambda session: session.list_tools()))

    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}},
        }
        for tool in result.tools
    ]


def search_jobs_via_mcp(
    query: str,
    seniority: str = "",
    employment_type: str = "",
    country: str = "",
    count: int = 20,
) -> list[dict[str, Any]]:
    """Call the search_jobs tool on the MCP server."""
    arguments = {
        "query": query,
        "seniority": seniority,
        "employment_type": employment_type,
        "country": country,
        "count": count,
    }
    result = asyncio.run(
        _run(lambda session: session.call_tool("search_jobs", arguments=arguments))
    )

    content = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )

    if not content:
        return []

    if isinstance(content, dict):
        return content.get("result", [content])

    return content if isinstance(content, list) else []
