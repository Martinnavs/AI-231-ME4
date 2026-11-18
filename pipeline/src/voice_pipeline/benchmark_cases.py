"""Case schema, strict JSONL loader, and helpers for the agent benchmark.

See ``docs/AGENT-BENCHMARK.md`` for the shared contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from voice_pipeline import evaluators

Stratum = Literal["regression", "hard", "happy", "false_premise"]
Role = Literal["user", "assistant"]
ToolResult = str | list[Any] | dict[str, Any]

STRATA: frozenset[str] = frozenset({"regression", "hard", "happy", "false_premise"})
ROLES: frozenset[str] = frozenset({"user", "assistant"})

_CASE_REQUIRED = frozenset({"id", "stratum", "provenance", "turns", "expect"})
_CASE_ALLOWED = _CASE_REQUIRED | frozenset({"known_failing", "offline", "note"})
_TURN_KEYS = frozenset({"role", "content"})
_EXPECT_ALLOWED = frozenset(
    {
        "required_tools",
        "forbidden_tools",
        "changeable",
        "abstain",
        "passages",
        "must",
        "false_premise",
    }
)
_OFFLINE_REQUIRED = frozenset({"model"})
_OFFLINE_ALLOWED = frozenset({"hedged", "router", "model", "tools", "stamp"})
_ROUTER_ALLOWED = frozenset({"lines", "reason", "delay_s"})
_MODEL_REQUIRED = frozenset({"text"})
_MODEL_ALLOWED = frozenset({"calls", "text", "delay_s", "parallel"})
_CALL_REQUIRED = frozenset({"tool"})
_CALL_ALLOWED = frozenset({"tool", "args"})
_TOOL_REQUIRED = frozenset({"result"})
_TOOL_ALLOWED = frozenset({"result", "delay_s", "by_router", "ledger"})
_OVERRIDE_ALLOWED = frozenset({"result", "delay_s", "ledger"})


class CaseError(ValueError):
    """Raised when a benchmark case line fails schema or reference validation."""


@dataclass(frozen=True)
class Turn:
    """One user or assistant message in a case's conversation history."""

    role: Role
    content: str


@dataclass(frozen=True)
class Expect:
    """Expected behaviour for evaluators (``must`` and ``false_premise`` are v1-recorded only)."""

    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    changeable: bool | None = None
    abstain: bool | None = None
    passages: tuple[str, ...] = ()
    must: tuple[str, ...] = ()
    false_premise: bool | None = None


@dataclass(frozen=True)
class RouterScript:
    """Scripted ``LocalRouter`` output for an offline fixture."""

    lines: str | None = None
    reason: str | None = None
    delay_s: float = 0.0


@dataclass(frozen=True)
class ModelCall:
    """One scripted model tool call."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelScript:
    """Scripted model actions: ordered tool calls followed by spoken text."""

    calls: tuple[ModelCall, ...] = ()
    text: str = ""
    delay_s: float = 0.0
    parallel: bool = False


@dataclass(frozen=True)
class ToolOverride:
    """Router-specific overrides when ``tool_call_id == 'router'``."""

    result: ToolResult | None = None
    delay_s: float | None = None
    ledger: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class ToolScript:
    """Scripted fake tool behaviour and ledger side effects."""

    result: ToolResult
    delay_s: float = 0.0
    by_router: ToolOverride | None = None
    ledger: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class OfflineFixture:
    """Scripted server components for running a case offline."""

    model: ModelScript
    router: RouterScript | None = None
    tools: dict[str, ToolScript] = field(default_factory=dict)
    hedged: bool = True
    stamp: bool = True


@dataclass(frozen=True)
class Case:
    """One agent benchmark case loaded from JSON lines."""

    id: str
    stratum: Stratum
    provenance: str
    turns: tuple[Turn, ...]
    expect: Expect
    known_failing: tuple[str, ...] = ()
    offline: OfflineFixture | None = None
    note: str | None = None


def _fail(line_no: int, case_id: Any, msg: str) -> CaseError:
    return CaseError(f"line {line_no} (id={case_id!r}): {msg}")


def _check_keys(
    obj: dict[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    where: str,
    line_no: int,
    case_id: Any,
) -> None:
    keys = set(obj.keys())
    missing = sorted(required - keys)
    if missing:
        raise _fail(line_no, case_id, f"{where}: missing required key(s): {', '.join(missing)}")
    unknown = sorted(keys - allowed)
    if unknown:
        raise _fail(line_no, case_id, f"{where}: unknown key(s): {', '.join(unknown)}")


def _as_dict(value: Any, field_name: str, line_no: int, case_id: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(line_no, case_id, f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_nonempty_str(value: Any, field_name: str, line_no: int, case_id: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(line_no, case_id, f"{field_name} must be a non-empty string")
    return value


def _as_str(value: Any, field_name: str, line_no: int, case_id: Any) -> str:
    if not isinstance(value, str):
        raise _fail(line_no, case_id, f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _as_bool(value: Any, field_name: str, line_no: int, case_id: Any) -> bool:
    if not isinstance(value, bool):
        raise _fail(line_no, case_id, f"{field_name} must be a bool, got {type(value).__name__}")
    return value


def _as_delay(value: Any, field_name: str, line_no: int, case_id: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(line_no, case_id, f"{field_name} must be a number, got {type(value).__name__}")
    delay = float(value)
    if not math.isfinite(delay) or delay < 0.0:
        raise _fail(line_no, case_id, f"{field_name} must be finite and >= 0, got {value!r}")
    return delay


def _as_str_tuple(value: Any, field_name: str, line_no: int, case_id: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(
            line_no, case_id, f"{field_name} must be a list of strings, got {type(value).__name__}"
        )
    out: list[str] = []
    for idx, item in enumerate(value):
        out.append(_as_nonempty_str(item, f"{field_name}[{idx}]", line_no, case_id))
    return tuple(out)


def _as_tool_result(value: Any, field_name: str, line_no: int, case_id: Any) -> ToolResult:
    if isinstance(value, str | list | dict):
        return value
    raise _fail(
        line_no,
        case_id,
        f"{field_name} must be str, list, or object, got {type(value).__name__}",
    )


def _as_ledger(
    value: Any, field_name: str, line_no: int, case_id: Any
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise _fail(
            line_no, case_id, f"{field_name} must be a list of objects, got {type(value).__name__}"
        )
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        rows.append(_as_dict(item, f"{field_name}[{idx}]", line_no, case_id))
    return tuple(rows)


def _parse_turns(raw: Any, line_no: int, case_id: Any) -> tuple[Turn, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail(line_no, case_id, "turns must be a non-empty list")
    turns: list[Turn] = []
    for idx, item in enumerate(raw):
        where = f"turns[{idx}]"
        obj = _as_dict(item, where, line_no, case_id)
        _check_keys(
            obj,
            required=_TURN_KEYS,
            allowed=_TURN_KEYS,
            where=where,
            line_no=line_no,
            case_id=case_id,
        )
        role_raw = obj["role"]
        if not isinstance(role_raw, str) or role_raw not in ROLES:
            raise _fail(
                line_no,
                case_id,
                f"{where}.role must be one of {sorted(ROLES)}, got {role_raw!r}",
            )
        content = _as_nonempty_str(obj["content"], f"{where}.content", line_no, case_id)
        turns.append(Turn(role=cast(Role, role_raw), content=content))
    if turns[-1].role != "user":
        raise _fail(line_no, case_id, "turns must end with a 'user' turn")
    return tuple(turns)


def _parse_expect(raw: Any, line_no: int, case_id: Any) -> Expect:
    obj = _as_dict(raw, "expect", line_no, case_id)
    _check_keys(
        obj,
        required=frozenset(),
        allowed=_EXPECT_ALLOWED,
        where="expect",
        line_no=line_no,
        case_id=case_id,
    )
    return Expect(
        required_tools=(
            _as_str_tuple(obj["required_tools"], "expect.required_tools", line_no, case_id)
            if "required_tools" in obj
            else ()
        ),
        forbidden_tools=(
            _as_str_tuple(obj["forbidden_tools"], "expect.forbidden_tools", line_no, case_id)
            if "forbidden_tools" in obj
            else ()
        ),
        changeable=(
            _as_bool(obj["changeable"], "expect.changeable", line_no, case_id)
            if "changeable" in obj
            else None
        ),
        abstain=(
            _as_bool(obj["abstain"], "expect.abstain", line_no, case_id)
            if "abstain" in obj
            else None
        ),
        passages=(
            _as_str_tuple(obj["passages"], "expect.passages", line_no, case_id)
            if "passages" in obj
            else ()
        ),
        must=(_as_str_tuple(obj["must"], "expect.must", line_no, case_id) if "must" in obj else ()),
        false_premise=(
            _as_bool(obj["false_premise"], "expect.false_premise", line_no, case_id)
            if "false_premise" in obj
            else None
        ),
    )


def _parse_known_failing(raw: Any, line_no: int, case_id: Any) -> tuple[str, ...]:
    items = _as_str_tuple(raw, "known_failing", line_no, case_id)
    for idx, item in enumerate(items):
        if item.count(":") != 1:
            raise _fail(
                line_no,
                case_id,
                f"known_failing[{idx}] must be '<F-id>:<evaluator>', got {item!r}",
            )
        f_id, eval_name = item.split(":", 1)
        if not f_id.strip():
            raise _fail(
                line_no,
                case_id,
                f"known_failing[{idx}] has empty F-id in {item!r}",
            )
        if eval_name not in evaluators.EVALUATORS:
            raise _fail(
                line_no,
                case_id,
                f"known_failing[{idx}] references unknown evaluator {eval_name!r}; "
                f"expected one of {sorted(evaluators.EVALUATORS)}",
            )
    return items


def _parse_router(raw: Any, line_no: int, case_id: Any) -> RouterScript:
    obj = _as_dict(raw, "offline.router", line_no, case_id)
    _check_keys(
        obj,
        required=frozenset(),
        allowed=_ROUTER_ALLOWED,
        where="offline.router",
        line_no=line_no,
        case_id=case_id,
    )
    has_lines = "lines" in obj
    has_reason = "reason" in obj
    if has_lines == has_reason:
        raise _fail(
            line_no,
            case_id,
            "offline.router must specify exactly one of 'lines' or 'reason'",
        )
    lines = _as_str(obj["lines"], "offline.router.lines", line_no, case_id) if has_lines else None
    reason = (
        _as_nonempty_str(obj["reason"], "offline.router.reason", line_no, case_id)
        if has_reason
        else None
    )
    delay_s = (
        _as_delay(obj["delay_s"], "offline.router.delay_s", line_no, case_id)
        if "delay_s" in obj
        else 0.0
    )
    return RouterScript(lines=lines, reason=reason, delay_s=delay_s)


def _parse_model(raw: Any, line_no: int, case_id: Any) -> ModelScript:
    obj = _as_dict(raw, "offline.model", line_no, case_id)
    _check_keys(
        obj,
        required=_MODEL_REQUIRED,
        allowed=_MODEL_ALLOWED,
        where="offline.model",
        line_no=line_no,
        case_id=case_id,
    )
    calls: list[ModelCall] = []
    if "calls" in obj:
        raw_calls = obj["calls"]
        if not isinstance(raw_calls, list):
            raise _fail(
                line_no,
                case_id,
                f"offline.model.calls must be a list, got {type(raw_calls).__name__}",
            )
        for idx, item in enumerate(raw_calls):
            where = f"offline.model.calls[{idx}]"
            call_obj = _as_dict(item, where, line_no, case_id)
            _check_keys(
                call_obj,
                required=_CALL_REQUIRED,
                allowed=_CALL_ALLOWED,
                where=where,
                line_no=line_no,
                case_id=case_id,
            )
            tool = _as_nonempty_str(call_obj["tool"], f"{where}.tool", line_no, case_id)
            args = (
                _as_dict(call_obj["args"], f"{where}.args", line_no, case_id)
                if "args" in call_obj
                else {}
            )
            calls.append(ModelCall(tool=tool, args=args))
    text = _as_str(obj["text"], "offline.model.text", line_no, case_id)
    delay_s = (
        _as_delay(obj["delay_s"], "offline.model.delay_s", line_no, case_id)
        if "delay_s" in obj
        else 0.0
    )
    parallel = (
        _as_bool(obj["parallel"], "offline.model.parallel", line_no, case_id)
        if "parallel" in obj
        else False
    )
    return ModelScript(calls=tuple(calls), text=text, delay_s=delay_s, parallel=parallel)


def _parse_tool_override(raw: Any, where: str, line_no: int, case_id: Any) -> ToolOverride:
    obj = _as_dict(raw, where, line_no, case_id)
    _check_keys(
        obj,
        required=frozenset(),
        allowed=_OVERRIDE_ALLOWED,
        where=where,
        line_no=line_no,
        case_id=case_id,
    )
    return ToolOverride(
        result=(
            _as_tool_result(obj["result"], f"{where}.result", line_no, case_id)
            if "result" in obj
            else None
        ),
        delay_s=(
            _as_delay(obj["delay_s"], f"{where}.delay_s", line_no, case_id)
            if "delay_s" in obj
            else None
        ),
        ledger=(
            _as_ledger(obj["ledger"], f"{where}.ledger", line_no, case_id)
            if "ledger" in obj
            else None
        ),
    )


def _parse_tools(raw: Any, line_no: int, case_id: Any) -> dict[str, ToolScript]:
    obj = _as_dict(raw, "offline.tools", line_no, case_id)
    tools: dict[str, ToolScript] = {}
    for name, spec_raw in obj.items():
        tool_name = _as_nonempty_str(name, "offline.tools key", line_no, case_id)
        where = f"offline.tools.{tool_name}"
        spec = _as_dict(spec_raw, where, line_no, case_id)
        _check_keys(
            spec,
            required=_TOOL_REQUIRED,
            allowed=_TOOL_ALLOWED,
            where=where,
            line_no=line_no,
            case_id=case_id,
        )
        tools[tool_name] = ToolScript(
            result=_as_tool_result(spec["result"], f"{where}.result", line_no, case_id),
            delay_s=(
                _as_delay(spec["delay_s"], f"{where}.delay_s", line_no, case_id)
                if "delay_s" in spec
                else 0.0
            ),
            by_router=(
                _parse_tool_override(spec["by_router"], f"{where}.by_router", line_no, case_id)
                if "by_router" in spec
                else None
            ),
            ledger=(
                _as_ledger(spec["ledger"], f"{where}.ledger", line_no, case_id)
                if "ledger" in spec
                else ()
            ),
        )
    return tools


def _parse_offline(raw: Any, line_no: int, case_id: Any) -> OfflineFixture:
    obj = _as_dict(raw, "offline", line_no, case_id)
    _check_keys(
        obj,
        required=_OFFLINE_REQUIRED,
        allowed=_OFFLINE_ALLOWED,
        where="offline",
        line_no=line_no,
        case_id=case_id,
    )
    return OfflineFixture(
        model=_parse_model(obj["model"], line_no, case_id),
        router=_parse_router(obj["router"], line_no, case_id) if "router" in obj else None,
        tools=_parse_tools(obj["tools"], line_no, case_id) if "tools" in obj else {},
        hedged=_as_bool(obj["hedged"], "offline.hedged", line_no, case_id)
        if "hedged" in obj
        else True,
        stamp=_as_bool(obj["stamp"], "offline.stamp", line_no, case_id) if "stamp" in obj else True,
    )


def _parse_case(raw_line: str, line_no: int, seen_ids: dict[str, int]) -> Case:
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise _fail(line_no, "<unknown>", f"invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise _fail(
            line_no,
            "<unknown>",
            f"case must be a JSON object, got {type(parsed).__name__}",
        )

    err_id = parsed.get("id", "<unknown>")
    _check_keys(
        parsed,
        required=_CASE_REQUIRED,
        allowed=_CASE_ALLOWED,
        where="case",
        line_no=line_no,
        case_id=err_id,
    )

    case_id = _as_nonempty_str(parsed["id"], "id", line_no, err_id)
    if case_id in seen_ids:
        raise _fail(
            line_no,
            case_id,
            f"duplicate case id (first seen on line {seen_ids[case_id]})",
        )
    seen_ids[case_id] = line_no

    stratum_raw = parsed["stratum"]
    if not isinstance(stratum_raw, str) or stratum_raw not in STRATA:
        raise _fail(
            line_no,
            case_id,
            f"unknown stratum {stratum_raw!r}; expected one of {sorted(STRATA)}",
        )

    provenance = _as_nonempty_str(parsed["provenance"], "provenance", line_no, case_id)
    turns = _parse_turns(parsed["turns"], line_no, case_id)
    expect = _parse_expect(parsed["expect"], line_no, case_id)
    known_failing = (
        _parse_known_failing(parsed["known_failing"], line_no, case_id)
        if "known_failing" in parsed
        else ()
    )
    offline = _parse_offline(parsed["offline"], line_no, case_id) if "offline" in parsed else None
    note = _as_str(parsed["note"], "note", line_no, case_id) if "note" in parsed else None

    return Case(
        id=case_id,
        stratum=cast(Stratum, stratum_raw),
        provenance=provenance,
        turns=turns,
        expect=expect,
        known_failing=known_failing,
        offline=offline,
        note=note,
    )


def load_cases(path: str | Path) -> list[Case]:
    """Load and strictly validate benchmark cases from a JSON-lines file."""
    text = Path(path).read_text(encoding="utf-8")
    cases: list[Case] = []
    seen_ids: dict[str, int] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        cases.append(_parse_case(raw_line, line_no, seen_ids))
    return cases


def hard(case: Case) -> bool:
    """Return True for every stratum except ``happy``."""
    return case.stratum != "happy"


def expect_mapping(case: Case) -> dict[str, Any]:
    """Return the ``expect`` mapping in the shape ``evaluators`` reads."""
    e = case.expect
    out: dict[str, Any] = {}
    if e.required_tools:
        out["required_tools"] = list(e.required_tools)
    if e.forbidden_tools:
        out["forbidden_tools"] = list(e.forbidden_tools)
    if e.changeable is not None:
        out["changeable"] = e.changeable
    if e.abstain is not None:
        out["abstain"] = e.abstain
    if e.passages:
        out["passages"] = list(e.passages)
    return out
