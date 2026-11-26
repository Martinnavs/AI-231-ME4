"""Research model adapter contract; no Google API requests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voice_pipeline.research_adapter import gemini_research_request


@pytest.mark.asyncio
async def test_research_adapter_reuses_client_without_document_schema() -> None:
    generate = AsyncMock(return_value=SimpleNamespace(text='{"queries": ["official source"]}'))
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    request = gemini_research_request(client, "configured-model", max_output_tokens=321)

    result = await request("loaded skill and planning instructions", "task envelope")

    assert result == '{"queries": ["official source"]}'
    arguments = generate.call_args.kwargs
    assert arguments["model"] == "configured-model"
    assert arguments["contents"] == "task envelope"
    assert arguments["config"].system_instruction == "loaded skill and planning instructions"
    assert arguments["config"].max_output_tokens == 321
    assert arguments["config"].response_mime_type == "application/json"
    assert arguments["config"].response_json_schema is None


def test_research_adapter_rejects_invalid_token_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        gemini_research_request(None, "configured-model", max_output_tokens=0)
