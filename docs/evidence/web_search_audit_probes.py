"""Offline probes. Run with repo .venv and PYTHONPATH=<repo>/src."""
import asyncio
import json

import httpx
from fastmcp import Client
from voice_agent.mcp_servers import gemini_grounding as g
from voice_agent.mcp_servers import tavily_search as t


async def probe():
    # Prove an explicit non-answer ends the provider chain despite a usable fallback.
    seen = []
    def handle(request):
        seen.append(str(request.url))
        if str(request.url) == g.GEMINI_URL:
            body = json.loads(request.content)
            if "tools" not in body:
                return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"verdict": "not_found", "answer": ""})}]}}]})
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "The winner is unavailable and unconfirmed."}]}, "groundingMetadata": {"webSearchQueries": ["winner"], "groundingChunks": [{"web": {"title": "example.org", "uri": "https://example.org"}}]}}]})
        return httpx.Response(200, json={"answer": "Example Person won.", "results": [{"title": "Result", "url": "https://example.org/result", "content": "Example Person won."}]})
    server = t.build_server(httpx.MockTransport(handle), key="fake", gemini_key="fake", ping_every_s=None, grounding=True, ledger=g.Ledger(None))
    async with Client(server) as client:
        reply = await client.call_tool("web_answer", {"query": "Who won the latest pageant?"})
    said = reply.structured_content["result"]
    assert said == g.HEDGE
    assert t.TAVILY_URL not in seen
    print(json.dumps({"probe": "nonanswer_ends_fallback", "reply": said, "google_requests": len(seen), "tavily_requests": 0}))

    # Semantic scope can differ despite high bag-of-words overlap.
    a = "What is the latest full length episode of The Koolpals podcast on Spotify?"
    b = "What is the latest full length episode of The Koolpals podcast on YouTube?"
    assert t.same_question(t.content_words(a), t.content_words(b))
    print(json.dumps({"probe": "platform_cache_collision", "questions": [a, b], "same_question": True}))

    evidence = "Alpha is not eligible for the prize."
    assert g.supported("Alpha is eligible for the prize.", evidence)
    print(json.dumps({"probe": "negation_inversion", "supported": True, "evidence": evidence}))

    data = {"results": [{"title": "Official result", "url": "https://example.org", "content": "Background " * 70 + "Example Person won."}]}
    text = t.research_text(data)
    assert "Example Person won" not in text
    print(json.dumps({"probe": "snippet_cutoff", "winner_in_input": True, "winner_in_output": False}))


asyncio.run(probe())
