"""MCP server that exposes the job-search tool (Week 4, Lesson 3)."""

from typing import Literal

try:
    from mcp.server import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from tools.job_search import search_jobs_api


mcp = MCPServer("career-job-search")


@mcp.tool(
    description=(
        "Search live remote jobs on Himalayas. "
        "query: a short job title or skill, for example 'data analyst'. "
        "seniority: the career level, or empty for any level. "
        "employment_type: the kind of contract, or empty for any type. "
        "country: a country name in English, for example 'Germany', or empty for any country. "
        "count: how many listings to return."
    )
)
def search_jobs(
    query: str,
    seniority: Literal["", "Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"] = "",
    employment_type: Literal["", "Full Time", "Part Time", "Contractor", "Temporary", "Intern", "Volunteer", "Other"] = "",
    country: str = "",
    count: int = 20,
) -> list[dict]:
    return search_jobs_api(
        query=query,
        seniority=seniority,
        employment_type=employment_type,
        country=country,
        count=count,
    )


if __name__ == "__main__":
    mcp.run()
