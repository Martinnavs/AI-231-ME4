"""Prefer IPv4 for name lookups.

Some networks (the Pi on the test network) have an IPv6 address but a broken IPv6 route: a TCP
connect to the Gemini API over IPv6 took 8.2 s against 0.09 s over IPv4, so the online probe
(2 s connect) marked the agent offline and every turn answered "I can't reach the internet".
`prefer_ipv4` makes lookups that allow both families return the IPv4 addresses only, falling back
to whatever the resolver gave when a host has no IPv4 address. `VOICE_AGENT_IPV6=1` turns it off.
"""

from __future__ import annotations

import os
import socket
from typing import Any

_installed = False


def prefer_ipv4() -> bool:
    """Install the filter once; returns True when it is active."""
    global _installed  # noqa: PLW0603
    if _installed:
        return True
    if os.environ.get("VOICE_AGENT_IPV6") == "1":
        return False
    original = socket.getaddrinfo

    def getaddrinfo(host: Any, port: Any, family: int = 0, *args: Any, **kwargs: Any) -> Any:
        found = original(host, port, family, *args, **kwargs)
        if family in (0, socket.AF_UNSPEC):
            v4 = [r for r in found if r[0] == socket.AF_INET]
            return v4 or found
        return found

    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    _installed = True
    return True
