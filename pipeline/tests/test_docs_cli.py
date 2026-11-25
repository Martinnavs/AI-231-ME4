"""The docs-first path is reachable from the command line: --docs-first and --docs-db."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_pipeline import agent_server, agent_tools


def test_docs_command_passes_the_prebuilt_db(tmp_path: Path) -> None:
    cmd = agent_tools.docs_command(tmp_path, {}, docs_db=tmp_path / "docs.db")
    assert cmd[cmd.index("--db") + 1] == str(tmp_path / "docs.db")
    assert cmd[cmd.index("--corpus") + 1] == str(tmp_path)


def test_docs_command_without_db_has_no_db_flag(tmp_path: Path) -> None:
    assert "--db" not in agent_tools.docs_command(tmp_path, {})


def test_main_passes_docs_first_and_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_run(*_args: object, **kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(agent_server, "run", fake_run)
    monkeypatch.setenv("LAN_SPEECH_TOKEN", "t")
    argv = ["--docs-first", "--docs-db", str(tmp_path / "d.db"), "--docs-dir", str(tmp_path)]
    assert agent_server.main(argv) == 0
    assert seen["docs_first"] is True
    assert seen["docs_db"] == str(tmp_path / "d.db")


def test_docs_first_defaults_off_and_follows_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_run(*_args: object, **kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(agent_server, "run", fake_run)
    monkeypatch.setenv("LAN_SPEECH_TOKEN", "t")
    monkeypatch.delenv("VOICE_AGENT_DOCS_FIRST", raising=False)
    agent_server.main([])
    assert seen["docs_first"] is False
