"""Tests for ``voice_pipeline.benchmark`` live guard, bearer client, and turn-id lookup."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from websockets.asyncio.server import serve

from conftest import LOOKED_UP, USER
from voice_pipeline import agent_server, agent_tools, bench_fakes, benchmark, dryrun, evaluators
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.benchmark_cases import ModelCall, RouterScript, ToolScript

SECRET_TOKEN = "live-test-token-do-not-print-9876"


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def _offline_happy_case(case_id: str = "happy-nvidia") -> dict[str, Any]:
    return {
        "id": case_id,
        "stratum": "happy",
        "provenance": "test",
        "turns": [{"role": "user", "content": USER}],
        "expect": {
            "required_tools": ["web_answer"],
            "forbidden_tools": ["search_docs"],
            "changeable": True,
            "abstain": False,
        },
        "offline": {
            "hedged": True,
            "stamp": True,
            "router": {"reason": "escalate for test"},
            "model": {
                "calls": [{"tool": "web_answer", "args": {"query": "ceo of nvidia"}}],
                "text": LOOKED_UP,
            },
            "tools": {
                "web_answer": {
                    "result": LOOKED_UP,
                    "ledger": [{"month": "2026-11", "searches": 1, "by": "model"}],
                }
            },
        },
    }


def _live_only_known_failing_case(case_id: str = "live-kf-case") -> dict[str, Any]:
    return {
        "id": case_id,
        "stratum": "regression",
        "provenance": "F2",
        "turns": [{"role": "user", "content": "Who is the current Prime Minister of Canada?"}],
        "expect": {
            "required_tools": ["web_answer"],
            "changeable": True,
            "abstain": False,
        },
        "known_failing": ["F2:note_reached"],
        "note": "Live-only case with known_failing tag.",
    }


def _build_fakes_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_stem: str = bench_fakes.RUN_STEM,
) -> tuple[AgentServer, Path, Path]:
    """Build an in-process ``AgentServer`` backed by 03b ``bench_fakes``."""
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    monkeypatch.setattr(agent_tools, "_web_answer", [])
    agent_tools.reset_turn_lookups()

    server_out = tmp_path / "agent_server_out"
    server_out.mkdir(parents=True, exist_ok=True)
    log_path = server_out / f"{run_stem}.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    record: list[dict[str, Any]] = []
    tool_script = ToolScript(
        result=LOOKED_UP,
        ledger=({"month": "2026-11", "searches": 1, "by": "model"},),
    )
    tools = ToolsSchema(
        standard_tools=[
            bench_fakes._make_tool_schema("web_answer", tool_script, ledger_path, record)
        ]
    )
    agent = bench_fakes._ScriptedRequest(
        tools,
        calls=(ModelCall(tool="web_answer", args={"query": "ceo of nvidia"}),),
        text=LOOKED_UP,
    )
    router = bench_fakes._FakeRouter(RouterScript(reason="escalate for test"))
    server = AgentServer(agent, record, log_path, tools, router)
    server.ledger = ledger_path
    server.stamp = lambda: "[Now: test]"
    return server, server_out, log_path


async def test_live_guard_without_yes_prints_estimate_and_opens_no_connection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    cases_file = _write_cases(
        tmp_path / "cases.jsonl",
        [
            _offline_happy_case("c1"),
            _offline_happy_case("c2"),
            _live_only_known_failing_case("c3"),
        ],
    )
    out_dir = tmp_path / "bench_out"
    connections = 0

    async def handler(ws: Any) -> None:
        nonlocal connections
        connections += 1
        await ws.close()

    def count_handshake(conn: Any, req: Any) -> None:
        del conn, req
        nonlocal connections
        connections += 1

    ws_server = await serve(handler, "127.0.0.1", 0, process_request=count_handshake)
    try:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        code = await asyncio.to_thread(
            benchmark.main,
            [
                "--cases",
                str(cases_file),
                "--out",
                str(out_dir),
                "--live",
                url,
                "--runs",
                "2",
                "--token-file",
                str(tmp_path / "nonexistent_token"),
            ],
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    assert code == 0
    assert connections == 0
    assert not out_dir.exists()

    out = capsys.readouterr().out
    low, high = dryrun.estimate_usd(6)
    assert "3 cases x 2 runs = 6 turns" in out
    assert f"~{int(6 * dryrun.REQUESTS_PER_TURN)} Gemini requests" in out
    assert "up to 6 grounded searches" in out
    assert f"${low:.4f}-${high:.4f}" in out
    assert "--yes" in out


async def test_live_end_to_end_against_in_process_agent_server_runs_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    cases_file = _write_cases(
        tmp_path / "cases.jsonl",
        [_offline_happy_case(), _live_only_known_failing_case()],
    )
    token_file = tmp_path / "token"
    token_file.write_text(f"{SECRET_TOKEN}\n", encoding="utf-8")
    out_dir = tmp_path / "bench_out"

    server, server_out, log_path = _build_fakes_server(tmp_path, monkeypatch)
    ws_server = await serve(
        server.handle,
        "127.0.0.1",
        0,
        process_request=agent_server.token_guard(SECRET_TOKEN),
    )
    try:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        code = await asyncio.to_thread(
            benchmark.main,
            [
                "--cases",
                str(cases_file),
                "--out",
                str(out_dir),
                "--live",
                url,
                "--runs",
                "2",
                "--yes",
                "--server-out",
                str(server_out),
                "--token-file",
                str(token_file),
            ],
        )
        if server.checks:
            await asyncio.gather(*server.checks, return_exceptions=True)
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        agent_tools.reset_turn_lookups()

    # Exit code is 0 even though known_failing (F2:note_reached) matches or differs in live mode.
    assert code == 0

    captured = capsys.readouterr()
    assert SECRET_TOKEN not in captured.out
    assert SECRET_TOKEN not in captured.err
    assert "live: model behaviour" in captured.out
    assert "2 cases x 2 runs = 4 turns" in captured.out
    assert "happy-nvidia: 2/2" in captured.out
    assert "live-kf-case:" in captured.out
    assert "composite" not in captured.out.lower()
    assert "aggregate" not in captured.out.lower()

    reports = sorted(out_dir.glob("*.json"))
    assert len(reports) == 1
    report_text = reports[0].read_text(encoding="utf-8")
    assert SECRET_TOKEN not in report_text

    report = json.loads(report_text)
    assert report["header"] == "live: model behaviour"
    assert report["mode"] == "live"
    assert report["runs"] == 2
    assert list(report["evaluators"]) == list(evaluators.EVALUATORS)
    forbidden_keys = {"aggregate", "composite", "score", "overall_score", "total_score"}
    assert not (forbidden_keys & set(report))
    assert not (forbidden_keys & set(report["evaluators"]))

    # Both offline-fixture case and live-only case ran twice each (4 total runs).
    assert [c["id"] for c in report["cases"]] == ["happy-nvidia", "live-kf-case"]
    happy_entry, kf_entry = report["cases"]
    assert happy_entry["runs"] == 2
    assert happy_entry["pass_count"] == 2
    assert len(happy_entry["turns"]) == 2
    assert kf_entry["runs"] == 2
    assert kf_entry["known_failing"] == ["F2:note_reached"]
    assert report["evaluators"]["lookup_required"]["overall"]["total"] == 4
    assert report["evaluators"]["lookup_required"]["overall"]["pass"] == 4

    # Server logged 4 turn rows with 2 distinct session UUIDs (one per run).
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    sessions = [r["session"] for r in rows]
    assert sessions[0] == sessions[1]
    assert sessions[2] == sessions[3]
    assert sessions[0] != sessions[2]


async def test_turn_id_row_lookup_retries_while_row_is_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    cases_file = _write_cases(tmp_path / "cases.jsonl", [_offline_happy_case("lookup-case")])
    token_file = tmp_path / "token"
    token_file.write_text(SECRET_TOKEN, encoding="utf-8")
    server_out = tmp_path / "server_out"
    server_out.mkdir()
    out_dir = tmp_path / "bench_out"

    run_id = "20261118-131500"
    target_turn = f"{run_id}-0002"
    log_path = server_out / f"{run_id}.jsonl"
    # Pre-populate an earlier row to verify exact turn-id matching (not just last line).
    log_path.write_text(
        json.dumps(
            {
                "turn": f"{run_id}-0001",
                "run": run_id,
                "question": "Old question",
                "calls": [],
                "cancelled": False,
                "route": "gemini",
                "tail": ["now"],
                "ledger": {"path": "", "from": 0, "to": 0, "n": 0, "rows": []},
                "status": "ok",
                "text": "Old answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    delayed_row = {
        "turn": target_turn,
        "run": run_id,
        "question": USER,
        "calls": [
            {
                "tool": "web_answer",
                "args": {"query": "ceo of nvidia"},
                "by": "model",
                "t0": 0.0,
                "s": 0.01,
                "status": "ok",
                "result_text": LOOKED_UP,
                "sources": ["CNN", "FT"],
            }
        ],
        "cancelled": False,
        "route": "routed",
        "tail": ["note", "now"],
        "ledger": {"path": "", "from": 0, "to": 0, "n": 1, "rows": [{"searches": 1}]},
        "status": "ok",
        "text": LOOKED_UP,
    }
    bg_tasks: list[asyncio.Task[None]] = []

    async def handler(ws: Any) -> None:
        await ws.recv()
        await ws.send(json.dumps({"text": LOOKED_UP}))
        await ws.send(json.dumps({"done": True, "turn": target_turn}))

        async def write_later() -> None:
            await asyncio.sleep(0.06)
            with log_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(delayed_row) + "\n")

        bg_tasks.append(asyncio.create_task(write_later()))

    ws_server = await serve(
        handler,
        "127.0.0.1",
        0,
        process_request=agent_server.token_guard(SECRET_TOKEN),
    )
    try:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        code = await asyncio.to_thread(
            benchmark.main,
            [
                "--cases",
                str(cases_file),
                "--out",
                str(out_dir),
                "--live",
                f"ws://127.0.0.1:{port}",
                "--runs",
                "1",
                "--yes",
                "--server-out",
                str(server_out),
                "--token-file",
                str(token_file),
            ],
        )
        if bg_tasks:
            await asyncio.gather(*bg_tasks)
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    assert code == 0
    report = json.loads(next(out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["cases"][0]["turns"] == [target_turn]
    assert report["cases"][0]["pass_count"] == 1


async def test_turn_id_lookup_timeout_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    monkeypatch.setattr(benchmark, "LIVE_ROW_TIMEOUT_S", 0.08)
    monkeypatch.setattr(benchmark, "LIVE_ROW_POLL_S", 0.01)

    cases_file = _write_cases(tmp_path / "cases.jsonl", [_offline_happy_case("missing-row")])
    token_file = tmp_path / "token"
    token_file.write_text(SECRET_TOKEN, encoding="utf-8")
    server_out = tmp_path / "server_out"
    server_out.mkdir()
    out_dir = tmp_path / "bench_out"

    async def handler(ws: Any) -> None:
        await ws.recv()
        await ws.send(json.dumps({"done": True, "turn": "20261118-131500-0099"}))

    ws_server = await serve(handler, "127.0.0.1", 0)
    try:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        code = await asyncio.to_thread(
            benchmark.main,
            [
                "--cases",
                str(cases_file),
                "--out",
                str(out_dir),
                "--live",
                f"ws://127.0.0.1:{port}",
                "--runs",
                "1",
                "--yes",
                "--server-out",
                str(server_out),
                "--token-file",
                str(token_file),
            ],
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    assert code == 2
    assert "not found" in capsys.readouterr().out
    assert not out_dir.exists()


@pytest.mark.parametrize("use_env", [False, True])
async def test_wrong_token_fails_cleanly_and_hides_tokens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    use_env: bool,
) -> None:
    wrong_token = "wrong-token-should-never-appear"
    cases_file = _write_cases(tmp_path / "cases.jsonl", [_offline_happy_case()])
    token_file = tmp_path / "token"
    if use_env:
        token_file.write_text(SECRET_TOKEN, encoding="utf-8")
        monkeypatch.setenv("LAN_SPEECH_TOKEN", wrong_token)
    else:
        monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
        token_file.write_text(wrong_token, encoding="utf-8")

    out_dir = tmp_path / "bench_out"
    server, server_out, log_path = _build_fakes_server(tmp_path, monkeypatch)
    ws_server = await serve(
        server.handle,
        "127.0.0.1",
        0,
        process_request=agent_server.token_guard(SECRET_TOKEN),
    )
    try:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        code = await asyncio.to_thread(
            benchmark.main,
            [
                "--cases",
                str(cases_file),
                "--out",
                str(out_dir),
                "--live",
                f"ws://127.0.0.1:{port}",
                "--runs",
                "1",
                "--yes",
                "--server-out",
                str(server_out),
                "--token-file",
                str(token_file),
            ],
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        agent_tools.reset_turn_lookups()

    assert code == 2
    assert not log_path.exists()
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert "401" in captured.out
    assert SECRET_TOKEN not in captured.out and SECRET_TOKEN not in captured.err
    assert wrong_token not in captured.out and wrong_token not in captured.err


def test_load_token_prefers_env_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n", encoding="utf-8")

    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    assert benchmark.load_token(token_file) == "file-token"

    monkeypatch.setenv("LAN_SPEECH_TOKEN", "  env-token \n")
    assert benchmark.load_token(token_file) == "env-token"

    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    assert benchmark.load_token(tmp_path / "missing") is None


def test_live_guard_with_default_cases_never_calls_connect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_connect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("connect must not be called without --yes")

    monkeypatch.setattr(benchmark, "connect", fail_connect)
    code = benchmark.main(["--live", "ws://127.0.0.1:8781", "--out", str(tmp_path / "out")])
    assert code == 0
    out = capsys.readouterr().out
    assert "39 cases x 5 runs = 195 turns" in out
    assert "487 Gemini requests" in out
    assert "up to 195 grounded searches" in out
    low, high = dryrun.estimate_usd(195)
    assert f"${low:.4f}-${high:.4f}" in out


def test_live_invalid_runs_and_empty_url_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases_file = _write_cases(tmp_path / "cases.jsonl", [_offline_happy_case()])
    assert (
        benchmark.main(["--cases", str(cases_file), "--live", "ws://127.0.0.1:1", "--runs", "0"])
        == 2
    )
    assert "--runs" in capsys.readouterr().out
    assert benchmark.main(["--cases", str(cases_file), "--live", ""]) == 2
    assert "--live" in capsys.readouterr().out
