import pytest


@pytest.fixture(autouse=True)
def _deterministic_docs_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "10")
    monkeypatch.setenv("VOICE_AGENT_DOCS_POLL_S", "0.2")
