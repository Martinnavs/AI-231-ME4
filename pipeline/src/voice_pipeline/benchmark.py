"""Offline and live agent benchmark runner CLI and JSON/table report.

    uv run python -m voice_pipeline.benchmark [--cases PATH] [--out DIR] [--only ID ...]
    uv run python -m voice_pipeline.benchmark --live ws://HOST:PORT [--runs N] [--yes]
        [--server-out DIR] [--token-file PATH]

Loads validated cases (``benchmark_cases``). Offline mode runs each case with an ``offline``
fixture through ``bench_fakes.run_offline_case`` in an isolated temporary directory. Live mode
prints the turn/request/search/cost estimate and exits 0 unless ``--yes`` is given; with ``--yes``,
it runs every case ``--runs`` times against the live server and reads turn rows from
``--server-out``. Both modes evaluate the T1 turn rows with ``evaluators.EVALUATORS``, print
per-evaluator pass rates (no composite score), and write ``<out>/<stamp>.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from voice_pipeline import bench_fakes, benchmark_cases, dryrun, evaluators
from voice_pipeline.benchmark_cases import Case, CaseError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "config/eval/benchmark_v1.jsonl"
DEFAULT_OUT = ROOT / "out/benchmark"
DEFAULT_SERVER_OUT = ROOT / "out/agent_server"
DEFAULT_TOKEN_FILE = Path.home() / ".config/lan_speech_token"
DEFAULT_RUNS = 5
LIVE_ROW_TIMEOUT_S = 5.0
LIVE_ROW_POLL_S = 0.02
OFFLINE_HEADER = "offline: numbers test server code, not model behaviour"
LIVE_HEADER = "live: model behaviour"
STRATUM_ORDER: tuple[str, ...] = ("regression", "hard", "happy", "false_premise")


class _CaseRunError(RuntimeError):
    """Wraps an unexpected error from running or scoring one offline case."""

    def __init__(self, case_id: str, cause: Exception) -> None:
        super().__init__(f"case {case_id!r} failed: {cause}")
        self.case_id = case_id
        self.cause = cause


def _check_known_failing(case: Case, verdicts: Sequence[evaluators.Verdict]) -> dict[str, Any]:
    """Check that a ``known_failing`` case fails on exactly its declared evaluator(s)."""
    declared = list(case.known_failing)
    expected = sorted({entry.split(":", 1)[1] for entry in declared})
    failed = sorted(v.name for v in verdicts if v.outcome == "fail")
    if not declared:
        return {
            "declared": [],
            "expected_evaluators": [],
            "failed_evaluators": failed,
            "missing": [],
            "unexpected": [],
            "ok": True,
        }
    expected_set = set(expected)
    failed_set = set(failed)
    missing = sorted(expected_set - failed_set)
    unexpected = sorted(failed_set - expected_set)
    return {
        "declared": declared,
        "expected_evaluators": expected,
        "failed_evaluators": failed,
        "missing": missing,
        "unexpected": unexpected,
        "ok": not missing and not unexpected,
    }


def _rate_block(counts: Mapping[str, int]) -> dict[str, Any]:
    passed = counts["pass"]
    failed = counts["fail"]
    na = counts["n/a"]
    applicable = passed + failed
    return {
        "pass": passed,
        "fail": failed,
        "n/a": na,
        "total": applicable + na,
        "applicable": applicable,
        "pass_rate": (passed / applicable) if applicable > 0 else None,
    }


def _ordered_strata(case_results: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(item["stratum"]) for item in case_results}
    ordered = [s for s in STRATUM_ORDER if s in present]
    extras = sorted(present - set(STRATUM_ORDER))
    return ordered + extras


def _summarize_evaluators(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute per-evaluator pass/fail/n/a counts and rates, overall and by stratum."""
    strata = _ordered_strata(case_results)
    summary: dict[str, Any] = {}
    for eval_name in evaluators.EVALUATORS:
        overall = {"pass": 0, "fail": 0, "n/a": 0}
        by_stratum = {s: {"pass": 0, "fail": 0, "n/a": 0} for s in strata}
        for item in case_results:
            stratum = str(item["stratum"])
            for verdict in item["verdicts"]:
                if verdict["name"] != eval_name:
                    continue
                outcome = str(verdict["outcome"])
                overall[outcome] += 1
                by_stratum[stratum][outcome] += 1
        summary[eval_name] = {
            "overall": _rate_block(overall),
            "by_stratum": {s: _rate_block(by_stratum[s]) for s in strata},
        }
    return summary


def _fmt_rate(rate: float | None) -> str:
    return f"{rate * 100:5.1f}%" if rate is not None else "     -"


def _format_table(eval_summary: Mapping[str, Mapping[str, Any]]) -> str:
    """Format per-evaluator pass rates (overall and by stratum) without any composite score."""
    lines = [f"{'evaluator':<20} {'scope':<14} {'pass':>4} {'fail':>4} {'n/a':>4} {'rate':>7}"]
    for eval_name, stats in eval_summary.items():
        overall = stats["overall"]
        lines.append(
            f"{eval_name:<20} {'overall':<14} {overall['pass']:>4} {overall['fail']:>4} "
            f"{overall['n/a']:>4} {_fmt_rate(overall['pass_rate']):>7}"
        )
        for stratum, s_stats in stats["by_stratum"].items():
            lines.append(
                f"{'':<20} {stratum:<14} {s_stats['pass']:>4} {s_stats['fail']:>4} "
                f"{s_stats['n/a']:>4} {_fmt_rate(s_stats['pass_rate']):>7}"
            )
    return "\n".join(lines)


async def _run_offline_cases(offline_cases: Sequence[Case]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="voice_bench_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, case in enumerate(offline_cases):
            workdir = tmp_root / f"case_{idx:04d}"
            try:
                row = await bench_fakes.run_offline_case(case, workdir)
                verdicts = evaluators.evaluate_all(row, benchmark_cases.expect_mapping(case))
            except Exception as exc:
                raise _CaseRunError(case.id, exc) from exc
            results.append(
                {
                    "id": case.id,
                    "stratum": case.stratum,
                    "provenance": case.provenance,
                    "turn": row.get("turn"),
                    "verdicts": [
                        {"name": v.name, "outcome": v.outcome, "reason": v.reason} for v in verdicts
                    ],
                    "known_failing": _check_known_failing(case, verdicts),
                }
            )
    return results


def load_token(path: Path) -> str | None:
    """Return the bearer token from ``LAN_SPEECH_TOKEN`` or ``path``; never print or log it."""
    if (env := os.environ.get("LAN_SPEECH_TOKEN")) and (stripped := env.strip()):
        return stripped
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _live_plan_lines(cases: Sequence[Case], runs: int) -> list[str]:
    total_turns = len(cases) * runs
    est_requests = int(total_turns * dryrun.REQUESTS_PER_TURN)
    low_usd, high_usd = dryrun.estimate_usd(total_turns)
    return [
        f"live plan: {len(cases)} cases x {runs} runs = {total_turns} turns",
        (
            f"estimate: ~{est_requests} Gemini requests, "
            f"up to {total_turns} grounded searches, "
            f"${low_usd:.4f}-${high_usd:.4f}"
        ),
    ]


def _run_id_from_turn(turn_id: str) -> str:
    run_id, sep, seq = turn_id.rpartition("-")
    if not sep or not run_id or not seq.isdigit():
        raise ValueError(f"invalid turn id: {turn_id!r}")
    return run_id


def _find_turn_row(log_path: Path, turn_id: str) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("turn") == turn_id:
            return row
    return None


async def _wait_for_turn_row(
    server_out: Path,
    turn_id: str,
    *,
    timeout_s: float | None = None,
    poll_s: float | None = None,
) -> dict[str, Any]:
    """Wait briefly for ``turn_id`` to be flushed to ``<server_out>/<run>.jsonl``."""
    wait_s = LIVE_ROW_TIMEOUT_S if timeout_s is None else timeout_s
    step_s = LIVE_ROW_POLL_S if poll_s is None else poll_s
    run_id = _run_id_from_turn(turn_id)
    log_path = server_out / f"{run_id}.jsonl"
    deadline = time.monotonic() + wait_s
    while True:
        row = _find_turn_row(log_path, turn_id)
        if row is not None:
            return row
        if time.monotonic() >= deadline:
            raise TimeoutError(f"turn {turn_id!r} not found in {log_path} within {wait_s:.1f}s")
        await asyncio.sleep(step_s)


async def _run_live_turn(
    url: str,
    token: str | None,
    case: Case,
    session: str,
    server_out: Path,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    messages = [{"role": t.role, "content": t.content} for t in case.turns]
    done_frame: dict[str, Any] | None = None
    try:
        async with connect(url, additional_headers=headers, open_timeout=10, max_size=None) as ws:
            await ws.send(json.dumps({"messages": messages, "session": session}))
            async for raw in ws:
                frame = json.loads(raw)
                if "error" in frame:
                    raise RuntimeError(f"server returned error: {frame['error']}")
                if frame.get("done"):
                    done_frame = frame
                    break
    except InvalidStatus as exc:
        raise RuntimeError(
            f"server rejected connection (HTTP {exc.response.status_code}); check bearer token"
        ) from None
    if done_frame is None:
        raise RuntimeError("server closed connection before done event")
    turn_id = done_frame.get("turn")
    if not isinstance(turn_id, str) or not turn_id:
        raise RuntimeError("done event missing 'turn' id")
    return await _wait_for_turn_row(server_out, turn_id)


async def _run_live_cases(
    cases: Sequence[Case],
    *,
    url: str,
    runs: int,
    server_out: Path,
    token: str | None,
) -> list[dict[str, Any]]:
    per_case: dict[str, list[dict[str, Any]]] = {c.id: [] for c in cases}
    for run_idx in range(1, runs + 1):
        session = uuid.uuid4().hex
        for case in cases:
            try:
                row = await _run_live_turn(url, token, case, session, server_out)
                verdicts = evaluators.evaluate_all(row, benchmark_cases.expect_mapping(case))
            except Exception as exc:
                raise _CaseRunError(case.id, exc) from exc
            per_case[case.id].append(
                {
                    "run": run_idx,
                    "turn": row.get("turn"),
                    "verdicts": [
                        {"name": v.name, "outcome": v.outcome, "reason": v.reason} for v in verdicts
                    ],
                    "known_failing": _check_known_failing(case, verdicts),
                }
            )

    case_results: list[dict[str, Any]] = []
    for case in cases:
        run_entries = per_case[case.id]
        pass_count = sum(
            1 for entry in run_entries if all(v["outcome"] != "fail" for v in entry["verdicts"])
        )
        per_eval: dict[str, dict[str, int]] = {}
        for eval_name in evaluators.EVALUATORS:
            counts = {"pass": 0, "fail": 0, "n/a": 0}
            for entry in run_entries:
                for verdict in entry["verdicts"]:
                    if verdict["name"] == eval_name:
                        counts[str(verdict["outcome"])] += 1
            per_eval[eval_name] = counts
        case_results.append(
            {
                "id": case.id,
                "stratum": case.stratum,
                "provenance": case.provenance,
                "runs": runs,
                "pass_count": pass_count,
                "turns": [entry["turn"] for entry in run_entries],
                "per_evaluator": per_eval,
                "run_results": run_entries,
                "known_failing": list(case.known_failing),
            }
        )
    return case_results


def _write_report(out_dir: Path, stamp: str, report: Mapping[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{stamp}.json"
    suffix = 1
    while report_path.exists():
        report_path = out_dir / f"{stamp}-{suffix}.json"
        suffix += 1
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help=f"JSONL case file (default: {DEFAULT_CASES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Directory for JSON reports (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ID",
        help="Run only the named case ID(s)",
    )
    parser.add_argument(
        "--live",
        default=None,
        metavar="URL",
        help="Run live against ws://HOST:PORT (prints plan; requires --yes to connect)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        metavar="N",
        help=f"Runs per case in --live mode (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually connect and run turns in --live mode",
    )
    parser.add_argument(
        "--server-out",
        type=Path,
        default=DEFAULT_SERVER_OUT,
        metavar="DIR",
        help=f"Agent server log directory for --live turn lookup (default: {DEFAULT_SERVER_OUT})",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        metavar="PATH",
        help=f"Bearer token file when LAN_SPEECH_TOKEN is unset (default: {DEFAULT_TOKEN_FILE})",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_live(args: argparse.Namespace, cases: Sequence[Case], cases_path: Path) -> int:
    for line in _live_plan_lines(cases, args.runs):
        print(line)
    if not args.yes:
        print("dry guard: pass --yes to run live against the server")
        return 0

    token = load_token(args.token_file)
    try:
        live_case_results = asyncio.run(
            _run_live_cases(
                cases,
                url=args.live,
                runs=args.runs,
                server_out=args.server_out,
                token=token,
            )
        )
    except _CaseRunError as exc:
        print(f"error running {exc}")
        return 2
    except Exception as exc:
        print(f"error in benchmark harness: {exc}")
        return 2

    flat_run_items = [
        {"stratum": item["stratum"], "verdicts": run_entry["verdicts"]}
        for item in live_case_results
        for run_entry in item["run_results"]
    ]
    eval_summary = _summarize_evaluators(flat_run_items)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report: dict[str, Any] = {
        "header": LIVE_HEADER,
        "mode": "live",
        "stamp": stamp,
        "cases_path": str(cases_path),
        "url": args.live,
        "runs": args.runs,
        "cases": live_case_results,
        "evaluators": eval_summary,
    }
    try:
        report_path = _write_report(args.out, stamp, report)
    except OSError as exc:
        print(f"error writing report to {args.out}: {exc}")
        return 2

    print(LIVE_HEADER)
    print(f"cases: {len(cases)}, runs: {args.runs}")
    print(_format_table(eval_summary))
    print("per-case pass counts:")
    for item in live_case_results:
        kf_note = (
            f" [known_failing: {', '.join(item['known_failing'])}]" if item["known_failing"] else ""
        )
        print(f"  {item['id']}: {item['pass_count']}/{item['runs']}{kf_note}")
    print(f"report: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline or live agent benchmark, write a JSON report, and print evaluator rates."""
    args = _parse_args(argv)
    if args.live is not None and not str(args.live).strip():
        print("error: --live URL must be non-empty")
        return 2
    if args.runs < 1:
        print(f"error: --runs must be >= 1, got {args.runs}")
        return 2

    cases_path: Path = args.cases if args.cases is not None else DEFAULT_CASES

    if args.cases is None and not cases_path.exists():
        print(f"no default case file at {cases_path}; nothing to run")
        return 0
    if not cases_path.exists():
        print(f"error: case file not found: {cases_path}")
        return 2

    try:
        cases = benchmark_cases.load_cases(cases_path)
    except (CaseError, OSError, UnicodeDecodeError) as exc:
        print(f"error loading cases from {cases_path}: {exc}")
        return 2

    if args.only is not None:
        known_ids = {c.id for c in cases}
        missing_ids = [cid for cid in args.only if cid not in known_ids]
        if missing_ids:
            print(f"error: unknown case id(s) in --only: {', '.join(missing_ids)}")
            return 2
        wanted = set(args.only)
        cases = [c for c in cases if c.id in wanted]

    if args.live is not None:
        return _run_live(args, cases, cases_path)

    offline_cases = [c for c in cases if c.offline is not None]
    skipped_live_only = [c.id for c in cases if c.offline is None]

    try:
        case_results = asyncio.run(_run_offline_cases(offline_cases))
    except _CaseRunError as exc:
        print(f"error running {exc}")
        return 2
    except Exception as exc:
        print(f"error in benchmark harness: {exc}")
        return 2

    eval_summary = _summarize_evaluators(case_results)
    mismatches = [item for item in case_results if not item["known_failing"]["ok"]]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    report: dict[str, Any] = {
        "header": OFFLINE_HEADER,
        "mode": "offline",
        "stamp": stamp,
        "cases_path": str(cases_path),
        "skipped_live_only": skipped_live_only,
        "cases": case_results,
        "evaluators": eval_summary,
        "known_failing_ok": not mismatches,
    }

    try:
        report_path = _write_report(args.out, stamp, report)
    except OSError as exc:
        print(f"error writing report to {args.out}: {exc}")
        return 2

    print(OFFLINE_HEADER)
    print(f"cases: {len(offline_cases)} offline, {len(skipped_live_only)} skipped (live-only)")
    if skipped_live_only:
        print(f"skipped (live-only): {', '.join(skipped_live_only)}")
    print(_format_table(eval_summary))
    for item in mismatches:
        kf = item["known_failing"]
        print(
            f"known_failing mismatch for {item['id']}: "
            f"expected={kf['expected_evaluators']} failed={kf['failed_evaluators']} "
            f"missing={kf['missing']} unexpected={kf['unexpected']}"
        )
    print(f"report: {report_path}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
