"""Gemini with Google Search first, then Tavily, then Wikipedia: order, spend ledger, prompt."""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastmcp import Client

from voice_agent.mcp_servers import gemini_grounding, tavily_search
from voice_agent.mcp_servers.gemini_grounding import Ledger, parse
from voice_agent.mcp_servers.tavily_search import build_server

TAVILY = {"answer": "Tavily says Leonardo painted it.", "results": [{"content": "Mona Lisa"}]}
SEARCHED = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "**Isko Moreno** is the mayor.\nHe took office in 2025."}]
            },
            "groundingMetadata": {
                "webSearchQueries": ["mayor of manila"],
                "groundingChunks": [{"web": {"title": "Manila - Wikipedia"}}],
            },
        }
    ],
    "usageMetadata": {"totalTokenCount": 83},
}
SETTLED = {
    "candidates": [{"content": {"parts": [{"text": "Canberra is the capital of Australia."}]}}],
    "usageMetadata": {"totalTokenCount": 54},
}


class Both(httpx.AsyncBaseTransport):
    """Gemini and Tavily endpoints with scripted replies; records the requests."""

    def __init__(self, gemini: httpx.Response | Exception, delay: float = 0.0) -> None:
        self.gemini, self.delay = gemini, delay
        self.seen: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        url = str(request.url)
        if url.startswith(gemini_grounding.GEMINI_URL):
            await asyncio.sleep(self.delay)
            if isinstance(self.gemini, Exception):
                raise self.gemini
            return self.gemini
        if url.startswith(tavily_search.TAVILY_URL):
            return httpx.Response(200, json=TAVILY)
        return httpx.Response(404)

    def count(self, prefix: str) -> int:
        return len([r for r in self.seen if str(r.url).startswith(prefix)])


async def _ask(
    transport: Both,
    query: str = "Who is the mayor of Manila?",
    gemini_key: str | None = "fake-gemini",
    ledger: Ledger | None = None,
    key: str | None = "fake-tavily",
) -> str:
    server = build_server(
        transport, key, None, grounding=True, gemini_key=gemini_key,
        ledger=ledger or Ledger(None),
    )  # fmt: skip
    async with Client(server) as client:
        result = await client.call_tool("web_answer", {"query": query})
    return str(result.structured_content["result"])


def test_parse_takes_the_text_the_searches_and_the_page_titles() -> None:
    text, searches, titles, tokens = parse(SEARCHED)
    assert text == "Isko Moreno is the mayor. He took office in 2025."  # no markdown, one line
    assert (searches, titles, tokens) == (1, ["Manila - Wikipedia"], 83)
    assert parse(SETTLED)[1] == 0 and parse({})[0] == ""


async def test_gemini_answers_first_and_tavily_is_not_asked() -> None:
    transport = Both(httpx.Response(200, json=SEARCHED))
    text = await _ask(transport)
    assert text == (
        "As per Wikipedia (the only source I found), "
        "Isko Moreno is the mayor. He took office in 2025."
    )
    assert transport.count(tavily_search.TAVILY_URL) == 0


async def test_the_request_has_the_search_tool_todays_date_and_the_key_as_a_header() -> None:
    transport = Both(httpx.Response(200, json=SETTLED))
    await _ask(transport, "What is the capital of Australia?")
    request = transport.seen[0]
    body = json.loads(request.content)
    assert request.headers["x-goog-api-key"] == "fake-gemini" and "key=" not in str(request.url)
    assert body["tools"] == [{"google_search": {}}]
    today = f"{datetime.now(ZoneInfo(gemini_grounding.TIMEZONE)):%A, %B %-d, %Y}"
    system = body["systemInstruction"]["parts"][0]["text"]
    assert today in system and "needs a search result" in system and "Cebu City" in system
    assert body["contents"][0]["parts"][0]["text"] == "What is the capital of Australia?"


@pytest.mark.parametrize(
    "gemini",
    [httpx.Response(429), httpx.Response(200, json={"candidates": []}),
     httpx.Response(200, text="not json"), httpx.ConnectError("down")],
)  # fmt: skip
async def test_tavily_answers_when_gemini_does_not(gemini: httpx.Response | Exception) -> None:
    transport = Both(gemini)
    assert await _ask(transport, "Who painted the Mona Lisa?") == TAVILY["answer"]
    assert transport.count(gemini_grounding.GEMINI_URL) == 1


async def test_a_slow_gemini_is_dropped_for_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tavily_search, "GEMINI_TIMEOUT_S", 0.05)
    transport = Both(httpx.Response(200, json=SETTLED), delay=0.5)
    assert await _ask(transport, "Who painted the Mona Lisa?") == TAVILY["answer"]


async def test_without_a_gemini_key_tavily_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_grounding, "KEY_FILE", Path("/nonexistent/key"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    transport = Both(httpx.Response(200, json=SETTLED))
    assert await _ask(transport, "Who painted the Mona Lisa?", gemini_key=None) == TAVILY["answer"]
    assert transport.count(gemini_grounding.GEMINI_URL) == 0


async def test_geminis_answer_is_not_second_guessed_by_the_name_check() -> None:
    reply = {"candidates": [{"content": {"parts": [{"text": "That year has not happened yet."}]}}]}
    transport = Both(httpx.Response(200, json=reply))
    text = await _ask(transport, "Who won the 2087 Nobel Prize in Physics?")
    assert text == "That year has not happened yet."


async def test_every_call_is_in_the_ledger_and_the_cap_sends_the_next_to_tavily(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, cap=1)
    transport = Both(httpx.Response(200, json=SEARCHED))
    assert "Isko Moreno" in await _ask(transport, "Who is the mayor of Manila?", ledger=ledger)
    row = json.loads(path.read_text())
    assert (row["searches"], row["tokens"]) == (1, 83)
    assert row["query"] == "Who is the mayor of Manila?" and row["sources"] == [
        "Manila - Wikipedia"
    ]
    assert ledger.exhausted
    # the cap is reached: a different question goes straight to Tavily
    assert await _ask(transport, "Who painted the Mona Lisa?", ledger=ledger) == TAVILY["answer"]
    assert transport.count(gemini_grounding.GEMINI_URL) == 1


def test_the_ledger_counts_this_month_only_and_survives_a_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"month": "2026-10", "searches": 40}\n{"month": "2026-11", "searches": 7}\nnot json\n'
    )
    assert Ledger(path, month="2026-11").searches == 0  # a bad line: start from zero, not crash
    path.write_text('{"month": "2026-10", "searches": 40}\n{"month": "2026-11", "searches": 7}\n')
    assert Ledger(path, month="2026-11").searches == 7
    assert Ledger(tmp_path / "missing.jsonl").searches == 0
    unwritable = Ledger(tmp_path / "no" / "such" / "dir" / "x.jsonl")
    unwritable.path = tmp_path / "file" / "x.jsonl"
    (tmp_path / "file").write_text("a file, not a folder")
    unwritable.add(gemini_grounding.Grounded("t", 2, [], None, 0.1))  # must not raise
    assert unwritable.searches == 2


async def test_a_failed_grounded_call_is_a_costless_ledger_row(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, month="2026-11")
    await _ask(Both(httpx.ConnectError("down")), ledger=ledger)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["failed"] == "ConnectError" and row["searches"] == 0 and row["tokens"] == 0
    assert Ledger(path, month="2026-11").searches == 0


async def test_a_grounded_timeout_is_logged_as_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(tavily_search, "GEMINI_TIMEOUT_S", 0.05)
    await _ask(Both(httpx.Response(200), delay=1.0), ledger=Ledger(path, month="2026-11"))
    assert json.loads(path.read_text().splitlines()[0])["failed"] == "timeout"


# --- the search gate: a changeable question answered without a search --------------------------

ZERO = {
    "candidates": [{"content": {"parts": [{"text": "Jensen Huang is the CEO of Nvidia."}]}}],
    "usageMetadata": {"totalTokenCount": 60},
}
FORCED = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Jensen Huang is the CEO of Nvidia."}]},
            "groundingMetadata": {
                "webSearchQueries": ["nvidia ceo"],
                "groundingChunks": [{"web": {"title": "nvidia.com", "uri": "https://x/1"}}],
                "groundingSupports": [{"groundingChunkIndices": [0]}],
            },
        }
    ],
    "usageMetadata": {"totalTokenCount": 90},
}
CEO = "Who's the CEO of Nvidia?"


class Scripted(Both):
    """Gemini replies one after another (the last repeats); Tavily as in `Both`."""

    def __init__(self, *replies: httpx.Response, tavily: httpx.Response | None = None) -> None:
        super().__init__(replies[0])
        self.replies, self.tavily = list(replies), tavily

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(gemini_grounding.GEMINI_URL):
            self.seen.append(request)
            return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if self.tavily is not None and str(request.url).startswith(tavily_search.TAVILY_URL):
            self.seen.append(request)
            return self.tavily
        return await super().handle_async_request(request)


def test_changeable_is_true_for_roles_and_current_things_and_false_for_settled_facts() -> None:
    for question in (CEO, "Who is the president of the University of the Philippines?",
                     "Who won the last NBA championship?", "What is the latest version of Python?",
                     "Who is the mayor here?"):  # fmt: skip
        assert gemini_grounding.changeable(question), question
    for question in ("What is the capital of France?", "Who wrote Noli Me Tangere?",
                     "Who painted the Mona Lisa?", "Who is Rizal?"):  # fmt: skip
        assert not gemini_grounding.changeable(question), question


def test_publisher_says_a_site_or_a_page_title_in_words() -> None:
    names = {
        "nvidia.com": "Nvidia", "www.up.edu.ph": "UP", "en.wikipedia.org": "Wikipedia",
        "bbc.co.uk": "BBC",
        "intentsignal.ai": "Intentsignal", "unifygtm.com": "Unifygtm",
        "theguardian.com": "The Guardian", "abs-cbn.com": "ABS-CBN", "gmanetwork.com": "GMA News",
        "mining.com": "Mining", "onenews.ph": "Onenews", "upd.edu.ph": "UP Diliman",
        "Jensen Huang - Forbes": "Forbes", "Nvidia | Wikipedia": "Wikipedia",
    }  # fmt: skip
    for title, spoken in names.items():
        assert gemini_grounding.publisher(title) == spoken, title
    assert gemini_grounding.publisher("A very long page title with no site name") is None
    assert gemini_grounding.publisher("") is None
    assert "/" not in (gemini_grounding.publisher("Page - <b>Evil</b>/x") or "")


def test_the_hedge_and_the_source_sentence_cannot_look_like_a_failed_lookup() -> None:
    # a literal copy of voice_pipeline.agent_tools.FAILURE: a reply that matches it is treated as a
    # failed lookup and asked a second time, so neither may match
    failure = re.compile(r"couldn't find|did not answer|not connected|can't say|error", re.I)
    assert not failure.search(gemini_grounding.HEDGE)
    assert not failure.search(" Source: Nvidia.")


async def test_a_changeable_answer_from_memory_is_asked_again_with_the_search_instruction() -> None:
    transport = Scripted(httpx.Response(200, json=ZERO), httpx.Response(200, json=FORCED))
    text = await _ask(transport, CEO)
    assert text == "As per Nvidia (the only source I found), Jensen Huang is the CEO of Nvidia."
    assert transport.count(gemini_grounding.GEMINI_URL) == 2
    first, second = (json.loads(r.content)["systemInstruction"]["parts"][0]["text"]
                     for r in transport.seen)  # fmt: skip
    assert gemini_grounding.FORCE_SEARCH not in first and gemini_grounding.FORCE_SEARCH in second


async def test_a_searched_or_settled_answer_makes_one_request() -> None:
    searched = Scripted(httpx.Response(200, json=FORCED))
    await _ask(searched, CEO)
    settled = Scripted(httpx.Response(200, json=SETTLED))
    await _ask(settled, "What is the capital of Australia?")
    assert searched.count(gemini_grounding.GEMINI_URL) == 1
    assert settled.count(gemini_grounding.GEMINI_URL) == 1


async def test_two_unsearched_answers_go_to_tavily_with_its_source() -> None:
    tavily = {"answer": "Jensen Huang is the CEO of Nvidia.", "results": [{"content": "x",
              "url": "https://www.nvidia.com/en-us/about/"}]}  # fmt: skip
    transport = Scripted(httpx.Response(200, json=ZERO), tavily=httpx.Response(200, json=tavily))
    text = await _ask(transport, CEO)
    assert text == "As per Nvidia, Jensen Huang is the CEO of Nvidia."
    assert transport.count(gemini_grounding.GEMINI_URL) == 2
    assert transport.count(tavily_search.TAVILY_URL) == 1


async def test_when_nothing_can_check_it_the_agent_says_so_without_the_guess_and_never_caches() -> (
    None
):
    transport = Scripted(httpx.Response(200, json=ZERO), tavily=httpx.Response(500))
    server = build_server(
        transport, "fake-tavily", None, grounding=True, gemini_key="fake", ledger=Ledger(None)
    )
    async with Client(server) as client:
        first = await client.call_tool("web_answer", {"query": CEO})
        second = await client.call_tool("web_answer", {"query": CEO})
    for answer in (first, second):
        said = answer.structured_content["result"]
        assert said == gemini_grounding.HEDGE
        assert "Jensen" not in said and "CEO" not in said  # the memory answer is never spoken
    assert transport.count(gemini_grounding.GEMINI_URL) == 4  # the second call asked again


async def test_a_settled_unsearched_answer_is_still_cached(tmp_path: Path) -> None:
    transport = Scripted(httpx.Response(200, json=SETTLED))
    server = build_server(
        transport, "fake-tavily", None, grounding=True, gemini_key="fake", ledger=Ledger(None)
    )
    async with Client(server) as client:
        for _ in range(2):
            await client.call_tool("web_answer", {"query": "What is the capital of Australia?"})
    assert transport.count(gemini_grounding.GEMINI_URL) == 1


async def test_the_retry_is_logged_and_the_venue_reaches_the_prompt(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    transport = Scripted(httpx.Response(200, json=ZERO), httpx.Response(200, json=FORCED))
    server = build_server(
        transport, "fake-tavily", None, grounding=True, gemini_key="fake",
        ledger=Ledger(path), place="Mandaue",
    )  # fmt: skip
    async with Client(server) as client:
        await client.call_tool("web_answer", {"query": CEO})
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [(r["searches"], r.get("forced", False), r["query"]) for r in rows] == [
        (0, False, CEO), (1, True, CEO)
    ]  # fmt: skip
    assert "The user is in Mandaue" in json.loads(transport.seen[0].content)["systemInstruction"][
        "parts"][0]["text"]  # fmt: skip


def test_the_home_flag_is_read_from_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Stub:
        def run(self, **_: object) -> None: ...

    def fake_build(**kwargs: object) -> Stub:
        seen.update(kwargs)
        return Stub()

    monkeypatch.setattr(tavily_search, "build_server", fake_build)
    tavily_search.main(["--home", "Mandaue"])
    assert seen["place"] == "Mandaue"
    tavily_search.main([])
    assert seen["place"] == "Cebu City"


def test_the_capture_script_sends_nothing_without_yes(capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import capture_grounding  # noqa: PLC0415

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("a request was sent")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx.Client, "post", refuse)
        assert capture_grounding.main(["nvidia_ceo=Who is the CEO of Nvidia?"]) == 0
    assert "nothing sent" in capsys.readouterr().out


def test_the_capture_script_refuses_to_write_a_file_that_holds_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import capture_grounding  # noqa: PLC0415

    leak = {"candidates": [{"content": {"parts": [{"text": "key is sekrit-key"}]}}]}
    monkeypatch.setattr(gemini_grounding, "load_key", lambda: "sekrit-key")
    monkeypatch.setattr(gemini_grounding, "LEDGER", tmp_path / "l.jsonl")
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *_, **__: httpx.Response(200, json=leak, request=httpx.Request("POST", "http://x")),
    )
    assert capture_grounding.main(["a=Q", "--yes", "--out", str(tmp_path)]) == 1
    assert not list(tmp_path.glob("a.json"))


def _answer(*titles: str, lead: int = 0) -> gemini_grounding.Grounded:
    sources = [(t, f"https://x/{i}") for i, t in enumerate(titles)]
    return gemini_grounding.Grounded("t", 1, list(titles), None, 0.1, sources, lead)


def test_the_sentence_says_two_sites_one_site_or_a_failed_double_check() -> None:
    say = gemini_grounding.source_sentence
    assert say(_answer("nvidia.com", "forbes.com"), True) == " Sources: Nvidia and Forbes."
    assert say(_answer("nvidia.com", "forbes.com", lead=1), True) == " Sources: Forbes and Nvidia."
    assert (
        say(_answer("nvidia.com", "nvidia.com"), True) == " Source: Nvidia. Only one source found."
    )
    assert say(_answer("nvidia.com"), False) == " Source: Nvidia."  # a settled fact: no caveat
    assert say(_answer("a page with a long title and no site"), True) == ""
    assert say(_answer(), True) == ""
    failure = re.compile(r"couldn't find|did not answer|not connected|can't say|error", re.I)
    for case in (_answer("nvidia.com", "forbes.com"), _answer("nvidia.com")):
        assert not failure.search(say(case, True))


# --- the second-opinion check (web_verify) -------------------------------------------------------


def _fresh(text: str, searches: int = 1) -> httpx.Response:
    chunks = [
        {"web": {"title": "sec.gov", "uri": "https://x/1"}},
        {"web": {"title": "cnn.com", "uri": "https://x/2"}},
    ]
    meta = {"webSearchQueries": ["q"] * searches, "groundingChunks": chunks} if searches else {}
    candidate: dict[str, object] = {"content": {"parts": [{"text": text}]}}
    if meta:
        candidate["groundingMetadata"] = meta
    return httpx.Response(
        200, json={"candidates": [candidate], "usageMetadata": {"totalTokenCount": 99}}
    )


def _plain(text: str) -> httpx.Response:
    reply = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return httpx.Response(200, json=reply)


KRESS = "Colette Kress is the interim CEO of Nvidia."


async def _verify(
    transport: Both,
    tmp_path: Path,
    maximum: int = 100,
    question: str = CEO,
    claim: str = "Jensen Huang is the CEO of Nvidia.",
) -> str:
    server = build_server(
        transport, "fake-tavily", None, grounding=True, gemini_key="fake", ledger=Ledger(None),
        verify_max=maximum,
        verify_budget=gemini_grounding.VerifyBudget(tmp_path / "used.json", maximum),
        outcomes=tmp_path / "outcomes.jsonl",
    )  # fmt: skip
    async with Client(server) as client:
        result = await client.call_tool("web_verify", {"question": question, "claim": claim})
    return str(result.structured_content["result"])


def test_a_comparison_is_read_from_one_word_and_anything_else_is_nothing() -> None:
    parse_same = gemini_grounding.parse_same
    assert parse_same("SAME") is True and parse_same("same.") is True
    assert parse_same("DIFFERENT") is False and parse_same("They are different.") is False
    assert parse_same("Hard to say") is None and parse_same("") is None


async def test_a_second_look_that_differs_is_returned_with_its_sources(tmp_path: Path) -> None:
    transport = Scripted(_fresh(KRESS), _plain("DIFFERENT"))
    out = await _verify(transport, tmp_path)
    assert out == f"VERDICT: DIFFERS\nCORRECTION: As per SEC and CNN, {KRESS}"
    first, second = (json.loads(r.content) for r in transport.seen)
    assert gemini_grounding.FORCE_SEARCH in first["systemInstruction"]["parts"][0]["text"]
    assert first["contents"][0]["parts"][0]["text"] == CEO and "tools" in first  # asked fresh
    assert "tools" not in second  # the comparison has no search tool and costs almost nothing
    assert "Answer 1: Jensen Huang" in second["contents"][0]["parts"][0]["text"]
    row = json.loads((tmp_path / "outcomes.jsonl").read_text())
    assert row["verdict"] == "DIFFERS" and row["second"] == KRESS and row["searches"] == 1


async def test_agreement_is_supported_and_silent(tmp_path: Path) -> None:
    transport = Scripted(_fresh("Jensen Huang leads Nvidia."), _plain("SAME"))
    assert await _verify(transport, tmp_path) == "VERDICT: SUPPORTED"


async def test_a_second_look_that_did_not_search_counts_for_nothing(tmp_path: Path) -> None:
    transport = Scripted(_fresh(KRESS, searches=0), _plain("DIFFERENT"))
    assert await _verify(transport, tmp_path) == "VERDICT: UNVERIFIED"
    assert transport.count(gemini_grounding.GEMINI_URL) == 1  # no comparison of memory with memory
    assert json.loads((tmp_path / "used.json").read_text())["used"] == 1  # but the check was spent


async def test_an_unreadable_comparison_or_a_failed_call_overrides_nothing(tmp_path: Path) -> None:
    assert await _verify(Scripted(_fresh(KRESS), _plain("Hard to say")), tmp_path) == (
        "VERDICT: UNVERIFIED"
    )
    assert await _verify(Both(httpx.Response(500)), tmp_path) == "VERDICT: UNVERIFIED"


async def test_the_allowance_is_kept_in_a_file_and_counted_across_servers(tmp_path: Path) -> None:
    def transport() -> Scripted:
        return Scripted(_fresh("Jensen Huang leads Nvidia."), _plain("SAME"))

    assert await _verify(transport(), tmp_path, maximum=2) == "VERDICT: SUPPORTED"
    assert await _verify(transport(), tmp_path, maximum=2) == "VERDICT: SUPPORTED"
    spent = transport()
    third = await _verify(spent, tmp_path, maximum=2)  # a new server: the count is in the file
    assert third == "SKIP: the allowance of 2 checks is used"
    assert json.loads((tmp_path / "used.json").read_text()) == {"used": 2, "max": 2}
    assert spent.count(gemini_grounding.GEMINI_URL) == 0  # the skipped one spent nothing


async def test_checks_are_off_by_default_and_skip_settled_facts(tmp_path: Path) -> None:
    transport = Scripted(_fresh(KRESS), _plain("SAME"))
    assert await _verify(transport, tmp_path, maximum=0) == "SKIP: checks are off"
    settled = await _verify(transport, tmp_path, question="What is the capital of France?")
    assert settled == "SKIP: not a fact that changes"
    assert transport.count(gemini_grounding.GEMINI_URL) == 0
    assert not (tmp_path / "used.json").exists()  # nothing was spent


def test_the_source_leads_the_answer_and_a_plain_first_word_is_lowercased() -> None:
    attribute = gemini_grounding.attribute
    assert attribute("The CEO is Jensen Huang.", ["Nvidia", "CNN"]) == (
        "As per Nvidia and CNN, the CEO is Jensen Huang."
    )
    assert attribute("Jensen Huang resigned.", ["SEC"]) == "As per SEC, Jensen Huang resigned."
    assert attribute("It is 3.15.2.", ["Python"], only_one=True) == (
        "As per Python (the only source I found), it is 3.15.2."
    )
    assert attribute("Paris.", []) == "Paris." and attribute("", ["SEC"]) == ""
    failure = re.compile(r"couldn't find|did not answer|not connected|can't say|error", re.I)
    assert not failure.search(attribute("x.", ["Nvidia"], only_one=True))


class Conserving:
    """Tavily credits past the line: research is Google's first (tavily_budget.Credits)."""

    async def mode(self, client: httpx.AsyncClient) -> str:
        return "conserve"

    def spend(self, n: int) -> None:
        return None


async def test_a_research_result_ends_with_the_sources_the_server_can_speak(
    tmp_path: Path,
) -> None:
    transport = Scripted(_fresh("Huang resigned on November 12, SEC filings say."))
    server = build_server(
        transport, "fake-tavily", None, grounding=True, gemini_key="fake", ledger=Ledger(None),
        credits=Conserving(),  # type: ignore[arg-type]
    )  # fmt: skip
    async with Client(server) as client:
        result = await client.call_tool(
            "web_research", {"query": "Nvidia CEO news", "recent": True}
        )
    text = str(result.structured_content["result"])
    assert "Huang resigned on November 12" in text
    assert text.rstrip().endswith("Sources: SEC and CNN.")  # what the backstop reads and speaks
    assert "Pages used: sec.gov; cnn.com" in text


# --- the directness check: a web answer must answer the question (ticket 245) -------------------


ZORB = "Who won the 2031 Zorblat Cup?"  # settled in shape: not changeable, not a skip pattern
BLURB = (
    "Live coverage of the 2031 Zorblat Cup final is underway at the arena. The crowd is on its "
    "feet as the match enters the final minutes. The reigning titleholder entering the final was "
    "Zed."
)
WINNER = "Vex won the 2031 Zorblat Cup."  # short (<= 12 words): skipped when it is the reply
LONG = "The 2031 Zorblat Cup was won by Vex after a close final on June 3 at the arena."
LONG_HOWTO = (
    "Boil the water first and then add the tea leaves to the pot and let it steep for a few "
    "minutes before serving it."
)


def _grounded_reply(text: str, searches: int = 1, title: str = "zorblat.example") -> httpx.Response:
    """A grounded reply (with the search tool's metadata) for the answer calls."""
    chunks = [{"web": {"title": title, "uri": "https://x/1"}}]
    meta = {"webSearchQueries": ["q"] * searches, "groundingChunks": chunks} if searches else {}
    candidate: dict[str, object] = {"content": {"parts": [{"text": text}]}}
    if meta:
        candidate["groundingMetadata"] = meta
    return httpx.Response(
        200, json={"candidates": [candidate], "usageMetadata": {"totalTokenCount": 50}}
    )


def _check(verdict: str, answer: str = "") -> httpx.Response:
    """The judge's JSON reply for the directness check calls (no search tool)."""
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps({"verdict": verdict, "answer": answer})}]
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 20},
        },
    )


class Direct(httpx.AsyncBaseTransport):
    """Answer calls (with the search tool) and directness checks (without it) on one URL."""

    def __init__(
        self,
        answers: list[httpx.Response],
        checks: list[httpx.Response],
        check_delay: float = 0.0,
    ) -> None:
        self.answers = list(answers)
        self.checks = list(checks)
        self.check_delay = check_delay
        self.seen: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if str(request.url).startswith(gemini_grounding.GEMINI_URL):
            is_check = "tools" not in json.loads(request.content)
            pool = self.checks if is_check else self.answers
            item = pool.pop(0) if len(pool) > 1 else pool[0]
            if is_check:
                await asyncio.sleep(self.check_delay)
            return item
        return httpx.Response(404)

    def answer_calls(self) -> list[httpx.Request]:
        return [r for r in self.seen if "tools" in json.loads(r.content)]

    def check_calls(self) -> list[httpx.Request]:
        return [r for r in self.seen if "tools" not in json.loads(r.content)]


async def _zorb(
    answers: list[httpx.Response],
    checks: list[httpx.Response],
    ledger: Ledger | None = None,
    query: str = ZORB,
    check_delay: float = 0.0,
) -> tuple[str, Direct]:
    transport = Direct(answers, checks, check_delay=check_delay)
    text = await _ask(transport, query, ledger=ledger)
    return text, transport


async def test_the_check_keeps_a_reply_that_answers() -> None:
    text, transport = await _zorb([_grounded_reply(LONG)], [_check("answers")])
    assert text == "As per Example, " + LONG[0].lower() + LONG[1:]
    assert len(transport.answer_calls()) == 1 and len(transport.check_calls()) == 1


async def test_the_check_lifts_the_answer_out_of_a_dump() -> None:
    dump = (
        "The 2031 Zorblat Cup drew a large crowd at the arena. The final took place on June 3. "
        + WINNER
        + " beating Zed in the final."
    )
    text, transport = await _zorb([_grounded_reply(dump)], [_check("describes_only", WINNER)])
    assert text == f"As per Example, {WINNER}"
    assert len(transport.answer_calls()) == 1 and len(transport.check_calls()) == 1


async def test_the_pre_result_blurb_is_re_asked_and_the_winner_is_spoken(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    text, transport = await _zorb(
        [_grounded_reply(BLURB), _grounded_reply(WINNER)],
        [_check("describes_only", ""), _check("answers")],
        ledger=ledger,
    )
    assert text == f"As per Example, {WINNER}"
    assert "reigning titleholder" not in text  # the pre-result blurb is never spoken
    answers, checks = transport.answer_calls(), transport.check_calls()
    assert len(answers) == 2 and len(checks) == 2
    first, second = (json.loads(r.content) for r in answers)
    assert gemini_grounding.FORCE_SEARCH not in first["systemInstruction"]["parts"][0]["text"]
    assert gemini_grounding.FORCE_SEARCH in second["systemInstruction"]["parts"][0]["text"]
    assert second["contents"][0]["parts"][0]["text"] == ZORB + gemini_grounding.RETRY_SUFFIX
    rows = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    answer_rows = [r for r in rows if "check" not in r]
    assert [(r["searches"], r.get("forced", False)) for r in answer_rows] == [(1, False), (1, True)]
    assert [r.get("check") for r in rows if "check" in r] == [
        "directness:describes_only",
        "directness:answers",
    ]


async def test_when_even_the_re_ask_is_not_an_answer_the_reply_is_the_hedge() -> None:
    dump2 = "The result has not been announced yet. The organizers say it will be confirmed later."
    text, transport = await _zorb(
        [_grounded_reply(BLURB), _grounded_reply(dump2)],
        [_check("describes_only", ""), _check("not_found", "")],
    )
    assert text == gemini_grounding.HEDGE
    assert "reigning titleholder" not in text and "Zed" not in text  # never the dump
    assert len(transport.answer_calls()) == 2 and len(transport.check_calls()) == 2


async def test_a_re_ask_that_still_describes_is_spoken_not_hedged() -> None:
    dump2 = "The agency publishes an estimate each year. The latest one is 341 million people."
    text, transport = await _zorb(
        [_grounded_reply(BLURB), _grounded_reply(dump2)],
        [_check("describes_only", ""), _check("describes_only", "")],
    )
    assert text != gemini_grounding.HEDGE
    assert "341 million" in text
    assert len(transport.answer_calls()) == 2 and len(transport.check_calls()) == 2


@pytest.mark.parametrize(
    "bad_check",
    [
        httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "I cannot determine this."}]}}]},
        ),  # not JSON
        httpx.Response(500),  # an HTTP error
    ],
)  # fmt: skip
async def test_a_failed_directness_check_fails_open_to_the_original(
    bad_check: httpx.Response,
) -> None:
    text, transport = await _zorb([_grounded_reply(LONG)], [bad_check])
    assert text == "As per Example, " + LONG[0].lower() + LONG[1:]
    assert len(transport.check_calls()) == 1 and len(transport.answer_calls()) == 1  # no retry


async def test_a_slow_directness_check_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_grounding, "DIRECTNESS_TIMEOUT_S", 0.05)
    text, transport = await _zorb([_grounded_reply(LONG)], [_check("answers")], check_delay=0.5)
    assert text == "As per Example, " + LONG[0].lower() + LONG[1:]
    assert len(transport.check_calls()) == 1 and len(transport.answer_calls()) == 1


async def test_the_check_is_skipped_for_how_to_and_short_answers() -> None:
    # a how-to question: the check never runs, even for a long reply
    text, transport = await _zorb(
        [_grounded_reply(LONG_HOWTO)], [_check("answers")], query="How to make tea?"
    )
    assert text == f"As per Example, {LONG_HOWTO}"
    assert len(transport.check_calls()) == 0 and len(transport.answer_calls()) == 1
    # a short, unhedged reply: nothing to check
    short, transport = await _zorb([_grounded_reply("Vex won the cup.")], [_check("answers")])
    assert short == "As per Example, Vex won the cup."
    assert len(transport.check_calls()) == 0


async def test_an_exhausted_ledger_keeps_the_original_without_a_retry(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", cap=1)
    text, transport = await _zorb(
        [_grounded_reply(LONG)], [_check("describes_only", "")], ledger=ledger
    )
    assert (
        text == "As per Example, " + LONG[0].lower() + LONG[1:]
    )  # kept: the re-ask would pass the cap
    assert len(transport.answer_calls()) == 1 and len(transport.check_calls()) == 1
    rows = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [r.get("check") for r in rows if "check" in r] == ["directness:describes_only"]


async def test_the_check_request_has_no_search_tool_and_asks_for_json() -> None:
    _, transport = await _zorb([_grounded_reply(LONG)], [_check("answers")])
    body = json.loads(transport.check_calls()[0].content)
    assert "tools" not in body  # the judge searches nothing
    config = body["generationConfig"]
    assert config["temperature"] == 0 and config["responseMimeType"] == "application/json"
    assert config["thinkingConfig"]["thinkingLevel"] == "minimal"
    prompt = body["contents"][0]["parts"][0]["text"]
    assert prompt.startswith(f"Question: {ZORB}\nPassage: ") and LONG in prompt
    assert gemini_grounding.SYSTEM_DIRECTNESS in body["systemInstruction"]["parts"][0]["text"]
