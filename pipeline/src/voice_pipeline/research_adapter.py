"""Opt-in Gemini adapter for the research harness, separate from document schemas."""

from __future__ import annotations

from typing import Any

from voice_agent.research.runner import Model


def gemini_research_request(client: Any, model: str, *, max_output_tokens: int = 1500) -> Model:
    """Reuse a caller-owned client/key and model ID; closing it remains the caller's job.

    Do not pass ask_documents.gemini_json_request: its fixed answer schema is
    incompatible with research plans/claims. Provider token billing and spend
    reservation must be enforced by the caller's existing ledger adapter.
    """
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")

    async def request(system: str, content: str) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        response = await client.aio.models.generate_content(
            model=model,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                max_output_tokens=max_output_tokens,
            ),
        )
        return str(response.text or "")

    return request
