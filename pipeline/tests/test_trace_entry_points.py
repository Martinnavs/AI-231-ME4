"""Cross-cutting checks through the production entry points: `benchmark.main` / `python -m
voice_pipeline.benchmark`, and `AgentServer.handle` over a real local WebSocket."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from voice_pipeline import agent_tools, benchmark
from voice_pipeline.agent_server import AgentServer

PIPELINE = Path(__file__).resolve().parents[1]
EXPECTED_KNOWN_FAILING = {
    "reg-f3-duplicate-lookups": {"duplicate_lookups"},
    "reg-f14-joined-failure-from-memory": {"support"},
}


def _report(out_dir: Path) -> dict[str, Any]:
    reports = sorted(out_dir.glob("*.json"))
    assert len(reports) == 1
    data: dict[str, Any] = json.loads(reports[0].read_text(encoding="utf-8"))
    return data


def _failed(item: dict[str, Any]) -> set[str]:
    return {v["name"] for v in item["verdicts"] if v["outcome"] == "fail"}


def _spans(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(c["by"], c["tool"], c["status"]) for c in row["calls"]]


def _ledger_callers(world: Any) -> list[str]:
    lines = world.ledger.read_text().splitlines() if world.ledger.exists() else []
    return [json.loads(line)["by"] for line in lines]


@pytest.fixture
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every socket connect and refuse anything that is not loopback."""
    seen: list[Any] = []
    real = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        seen.append(address)
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"offline benchmark tried to reach {address!r}")
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    return seen


def test_benchmark_main_offline_on_the_shipped_cases_is_deterministic_and_loopback_only(
    tmp_path: Path, loopback_only: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = tmp_path / "run1", tmp_path / "run2"
    assert benchmark.main(["--out", str(first)]) == 0
    assert benchmark.main(["--out", str(second)]) == 0
    assert "offline: numbers test server code" in capsys.readouterr().out

    a, b = _report(first), _report(second)
    assert a["known_failing_ok"] is True and a["mode"] == "offline"
    assert a["evaluators"] == b["evaluators"]
    assert [(c["id"], c["verdicts"]) for c in a["cases"]] == [
        (c["id"], c["verdicts"]) for c in b["cases"]
    ]
    assert loopback_only, "the in-process server should have been reached over loopback"
    assert all(addr[0] in ("127.0.0.1", "::1") for addr in loopback_only)


def test_benchmark_main_known_failing_cases_fail_exactly_the_declared_evaluator(
    tmp_path: Path,
) -> None:
    assert benchmark.main(["--out", str(tmp_path)]) == 0
    by_id = {c["id"]: c for c in _report(tmp_path)["cases"]}
    for case_id, evaluators in EXPECTED_KNOWN_FAILING.items():
        assert _failed(by_id[case_id]) == evaluators, case_id
        assert by_id[case_id]["known_failing"]["ok"] is True
    for item in by_id.values():
        if item["id"] not in EXPECTED_KNOWN_FAILING and not item["known_failing"]["declared"]:
            assert _failed(item) == set(), item["id"]


def test_the_module_entry_point_runs_offline_in_a_fresh_process(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-m", "voice_pipeline.benchmark", "--out", str(tmp_path)],
        cwd=PIPELINE,
        env={"PYTHONPATH": str(PIPELINE / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout[-400:] + done.stderr[-400:]
    assert "offline: numbers test server code" in done.stdout
    assert _report(tmp_path)["known_failing_ok"] is True


def test_live_without_yes_never_connects_and_writes_no_report(
    tmp_path: Path, loopback_only: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code = benchmark.main(
        [
            "--live",
            "ws://127.0.0.1:9",
            "--token-file",
            str(tmp_path / "none"),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert code == 0
    assert loopback_only == []
    assert not (tmp_path / "o").exists()
    assert "--yes" in capsys.readouterr().out


async def test_handle_router_wins_trace_matches_the_calls_made(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=False)
    frames, row = await serve_turn(world.server)
    assert _spans(row) == [("router", "web_answer", "ok")]
    assert _ledger_callers(world) == ["router"]
    assert [r["by"] for r in row["ledger"]["rows"]] == _ledger_callers(world)
    assert frames[-1] == {"done": True, "turn": row["turn"]}
    assert row["hedge_winner"] == "router"
    assert [c["tool"] for c in world.record] == ["web_answer"]


async def test_handle_gemini_wins_cancels_the_router_lookup_and_only_model_searched(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=True)
    frames, row = await serve_turn(world.server)
    assert _spans(row) == [("router", "web_answer", "cancelled"), ("model", "web_answer", "ok")]
    assert _ledger_callers(world) == [
        "call-1"
    ]  # the scripted model request's tool_call_id; the router never wrote
    assert [r["by"] for r in row["ledger"]["rows"]] == [
        "call-1"
    ]  # the scripted model request's tool_call_id; the router never wrote
    assert frames[-1]["turn"] == row["turn"]
    assert row["tokens"]["requests"] == 2 and row["tokens"]["prompt"] == 180
    ok = [c for c in row["calls"] if c["status"] == "ok"]
    assert len(ok) == len(_ledger_callers(world))
    assert "result_text" not in row["calls"][0]


async def test_handle_fast_path_trace_and_the_tool_invoked_once_with_no_model_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    invoked: list[tuple[str, Any]] = []

    async def smart_home(params: Any) -> None:
        invoked.append((params.tool_call_id, dict(params.arguments)))
        await params.result_callback({"ok": True})

    schema = FunctionSchema(name="smart_home", description="d", properties={}, required=[],
                            handler=smart_home)  # fmt: skip
    record: list[dict[str, Any]] = []
    tools = ToolsSchema(standard_tools=[agent_tools._recorded(schema, record)])

    class NoModel:
        asked = 0
        tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

        async def ask(self, *_: Any, **__: Any) -> dict[str, Any]:
            NoModel.asked += 1
            raise AssertionError("the fast path must not reach the model")

    server = AgentServer(NoModel(), record, tmp_path / "20261118-130000.jsonl", tools, None)  # type: ignore[arg-type]
    server.ledger = tmp_path / "ledger.jsonl"
    frames, row = await serve_turn(server, "turn on the lights")

    assert NoModel.asked == 0
    assert row["route"] == "fast" and row["fast_path"] is True
    assert _spans(row) == [("fast-path", "smart_home", "ok")]
    assert [by for by, _ in invoked] == ["fast-path"] and len(invoked) == len(row["calls"])
    assert row["tokens"] == {"requests": 0, "prompt": 0, "completion": 0, "cached": 0}
    assert row["ledger"]["rows"] == []
    assert frames[-1] == {"done": True, "turn": row["turn"]}
