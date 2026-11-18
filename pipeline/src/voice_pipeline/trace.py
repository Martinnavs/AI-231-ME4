"""Helpers for the per-turn trace row (docs/TURN-TRACE.md): ids, spans, tokens, ledger slice."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

RESULT_CAP = 2000  # characters of a lookup's result kept in its span
SESSION_CAP = 64
LEDGER_ROWS_CAP = 10
SERVER_CALLERS = frozenset({"router", "fast-path"})  # tool_call_ids the server itself uses

_started = time.perf_counter()


def begin_turn() -> None:
    global _started
    _started = time.perf_counter()


def elapsed() -> float:
    """Seconds since the turn started."""
    return time.perf_counter() - _started


def caller(tool_call_id: Any) -> str:
    return str(tool_call_id) if tool_call_id in SERVER_CALLERS else "model"


def clean_session(value: Any) -> str | None:
    """The client's session id, which is untrusted: a non-empty string, cut to SESSION_CAP."""
    if isinstance(value, str) and value:
        return value[:SESSION_CAP]
    return None


def turn_id(run: str, n: int) -> str:
    return f"{run}-{n:04d}"


def text_of(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def annotate(
    entry: dict[str, Any],
    *,
    by: str,
    t0: float,
    status: str,
    results: list[Any],
    keep_text: bool = False,
    sources: tuple[list[str], bool] | None = None,
) -> None:
    """Add the span fields to a call's record entry (existing keys are left alone)."""
    entry.update(by=by, t0=round(t0, 3), s=round(elapsed() - t0, 3), status=status)
    if not results:
        return
    text = text_of(results[0])
    entry["result_sha"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    entry["result_chars"] = len(text)
    if keep_text:
        entry["result_text"] = text[:RESULT_CAP]
    if sources is not None:
        entry["sources"] = sources[0]
        entry["only_one_source"] = sources[1]


def stamp_new(record: list[dict[str, Any]], mark: int, by: str, t0: float, status: str) -> None:
    """Span fields for entries a tool appended itself: smart_home writes an `intent` entry with no
    `tool` key and no wrapper of its own to time it (entries with a `tool` time themselves)."""
    for entry in record[mark:]:
        if "by" not in entry and "tool" not in entry:
            entry.update(by=by, t0=round(t0, 3), s=round(elapsed() - t0, 3), status=status)


def ledger_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def ledger_slice(path: Path, start: int, end: int) -> dict[str, Any]:
    """The ledger rows appended between two byte offsets (at most LEDGER_ROWS_CAP are kept)."""
    rows: list[Any] = []
    if end > start:
        try:
            with path.open("rb") as source:
                source.seek(start)
                chunk = source.read(end - start)
        except OSError:
            chunk = b""
        for line in chunk.decode(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return {
        "path": str(path),
        "from": start,
        "to": end,
        "n": len(rows),
        "rows": rows[:LEDGER_ROWS_CAP],
    }


def usage_total(agent: Any) -> dict[str, int]:
    """LLM requests and tokens counted so far over every request of `agent` (hedge losers too)."""
    total = {"requests": 0, "prompt": 0, "completion": 0, "cached": 0}
    for one in getattr(agent, "agents", None) or [agent]:
        usage = getattr(getattr(one, "tap", None), "usage", None)
        if isinstance(usage, dict):
            for key in total:
                total[key] += int(usage.get(key, 0))
    return total


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}
