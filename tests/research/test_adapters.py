from __future__ import annotations

from typing import Any

import pytest

from voice_agent.agent.mcp_clients import ToolOutcome, ToolSpec
from voice_agent.research.adapters import McpSearch


class FakeRegistry:
    def __init__(self, *, tool: str = "web_research", result: ToolOutcome) -> None:
        self.tools = [ToolSpec("web__web_research", "web", tool, "search", {})]
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        self.calls.append((name, arguments))
        return self.result


@pytest.mark.asyncio
async def test_parses_actual_untrusted_search_text_format_with_optional_date() -> None:
    text = (
        "Searched on 2026-11-26 (Asia/Manila).\n"
        "<untrusted_search_data>\n"
        "Official report | https://example.org/report | Results say A | and B. | 2026-11-25\n"
        "Dataset page | https://example.org/data | Download the current dataset.\n"
        "</untrusted_search_data>"
    )
    registry = FakeRegistry(result=ToolOutcome(text, False, 0.01))
    search = McpSearch(registry, recent=True)  # type: ignore[arg-type]

    sources = await search("official report query")

    assert [(s.title, s.url, s.text, s.published_at) for s in sources] == [
        (
            "Official report",
            "https://example.org/report",
            "Results say A | and B.",
            "2026-11-25",
        ),
        ("Dataset page", "https://example.org/data", "Download the current dataset.", None),
    ]
    assert registry.calls == [
        ("web__web_research", {"query": "official report query", "recent": True})
    ]


def test_refuses_configured_non_research_tool() -> None:
    registry = FakeRegistry(tool="web_answer", result=ToolOutcome("answer", False, 0.01))

    with pytest.raises(ValueError, match="web_research MCP tool is required"):
        McpSearch(registry)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ToolOutcome("search failed", True, 0.01),
        ToolOutcome("one result\n[truncated 75 chars]", False, 0.01),
    ],
)
async def test_rejects_tool_errors_and_truncated_results(outcome: ToolOutcome) -> None:
    search = McpSearch(FakeRegistry(result=outcome))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="search failed or was truncated"):
        await search("some question")


@pytest.mark.asyncio
async def test_ignores_non_result_lines_and_returns_empty_for_no_sources() -> None:
    text = "Searched on today.\n<untrusted_search_data>\nmalformed line\n</untrusted_search_data>"
    search = McpSearch(FakeRegistry(result=ToolOutcome(text, False, 0.01)))  # type: ignore[arg-type]

    assert await search("some question") == []
