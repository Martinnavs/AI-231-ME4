"""Adapters to the existing MCP research service; no new search provider."""

from __future__ import annotations

import re
from datetime import datetime

from voice_agent.agent.mcp_clients import ToolRegistry
from voice_agent.research.runner import Source


class McpSearch:
    """Read search snippets, without falling back to a builder or a fact-answer tool.

    One adapter invocation can cause multiple backend queries/retries in the
    existing server. Its own credit policy remains responsible for those calls.
    """

    def __init__(
        self, tools: ToolRegistry, tool: str = "web__web_research", *, recent: bool = False
    ) -> None:
        if not any(spec.name == tool and spec.tool == "web_research" for spec in tools.tools):
            raise ValueError("a configured web_research MCP tool is required")
        self.tools, self.tool, self.recent = tools, tool, recent

    async def __call__(self, query: str) -> list[Source]:
        result = await self.tools.call(self.tool, {"query": query, "recent": self.recent})
        if result.is_error or "[truncated " in result.text:
            raise ValueError("research search failed or was truncated")
        sources = []
        for line in result.text.splitlines():
            match = re.fullmatch(r"(.*?) \| (https?://[^\s|]+) \| (.*)", line)
            if match is None:
                continue
            title, url, text = match.groups()
            published = None
            snippet, separator, tail = text.rpartition(" | ")
            if separator and re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T[^|]+)?", tail):
                try:
                    datetime.fromisoformat(tail)
                except ValueError:
                    pass
                else:
                    text, published = snippet, tail
            sources.append(Source(title, url, text, published))
        return sources
