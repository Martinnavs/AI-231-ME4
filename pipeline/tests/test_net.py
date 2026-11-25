"""prefer_ipv4 keeps IPv4 results and falls back when a host has none."""

from __future__ import annotations

import socket

import pytest

from voice_pipeline import net


def _fake(results: list[tuple[int, int, int, str, tuple[str, int]]]):
    def fake(host, port, family=0, *args, **kwargs):  # type: ignore[no-untyped-def]
        return [r for r in results if family in (0, r[0])]

    return fake


V4 = (socket.AF_INET, 1, 6, "", ("1.2.3.4", 443))
V6 = (socket.AF_INET6, 1, 6, "", ("::1", 443, 0, 0))


@pytest.fixture
def restore(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    original = socket.getaddrinfo
    monkeypatch.setattr(net, "_installed", False)
    monkeypatch.delenv("VOICE_AGENT_IPV6", raising=False)
    yield
    socket.getaddrinfo = original  # type: ignore[assignment]


def test_drops_ipv6_when_ipv4_exists(restore, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(socket, "getaddrinfo", _fake([V6, V4]))
    assert net.prefer_ipv4()
    assert socket.getaddrinfo("h", 443) == [V4]


def test_keeps_ipv6_when_host_has_only_ipv6(restore, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(socket, "getaddrinfo", _fake([V6]))
    net.prefer_ipv4()
    assert socket.getaddrinfo("h", 443) == [V6]


def test_off_switch(restore, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VOICE_AGENT_IPV6", "1")
    assert net.prefer_ipv4() is False
