#!/usr/bin/env python3
"""Paired significance comparison for two per-query RAG evaluation JSONL files.

Pure Python standard library only. Joins rows by ``id``, reports counts,
wins/losses ``(b, c)``, exact two-sided McNemar p-value via ``math.comb``,
Wilson 95% confidence intervals per arm, delta, and flipped query IDs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_METRICS: tuple[str, ...] = ("hit3", "hit4", "hit5")
DEFAULT_WILSON_Z: float = 1.959963984540054


@dataclass(frozen=True)
class PairedComparison:
    """Result of a paired binary comparison between baseline A and candidate B."""

    metric: str
    n: int
    hits_a: int
    hits_b: int
    b_wins: int
    c_losses: int
    delta: float
    p_value: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    flipped_wins: tuple[str, ...]
    flipped_losses: tuple[str, ...]
    only_in_a: tuple[str, ...]
    only_in_b: tuple[str, ...]
    verdict: str


def _parse_hit_value(value: Any, *, row_id: str, metric: str, source: Path) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"{source}: row {row_id!r} field {metric!r} must be bool or 0/1, got {value!r}"
    )


def load_per_query_hits(path: Path | str, metric: str) -> dict[str, bool]:
    """Load ``id -> hit`` mapping for ``metric`` from a ``--per-query`` JSONL file."""
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"unsupported metric {metric!r}: expected one of {ALLOWED_METRICS}")
    file_path = Path(path)
    hits: dict[str, bool] = {}
    for line_no, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"{file_path}:{line_no}: expected JSON object")
        raw_id = parsed.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"{file_path}:{line_no}: missing or invalid 'id'")
        row_id = raw_id.strip()
        if row_id in hits:
            raise ValueError(f"{file_path}:{line_no}: duplicate id {row_id!r}")
        if metric not in parsed:
            raise ValueError(f"{file_path}:{line_no}: row {row_id!r} missing {metric!r}")
        hits[row_id] = _parse_hit_value(
            parsed[metric],
            row_id=row_id,
            metric=metric,
            source=file_path,
        )
    return hits


def wilson_ci(hits: int, n: int, z: float = DEFAULT_WILSON_Z) -> tuple[float, float]:
    """Compute the two-sided Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    if hits < 0 or hits > n:
        raise ValueError(f"hits ({hits}) must be in [0, {n}]")
    p = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Compute the exact two-sided McNemar p-value from discordant counts ``(b, c)``."""
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail_sum = sum(math.comb(m, i) for i in range(k + 1))
    if m <= 1000:
        return min(1.0, (2 * tail_sum) / (1 << m))
    shift = m - 500
    return min(1.0, (2 * (tail_sum >> shift)) / (1 << 500))


def compare_per_query(
    a_hits: Mapping[str, bool],
    b_hits: Mapping[str, bool],
    *,
    metric: str = "hit4",
) -> PairedComparison:
    """Compare paired binary hit maps ``a_hits`` (A) and ``b_hits`` (B) joined by ID."""
    joined = [qid for qid in a_hits if qid in b_hits]
    only_a = tuple(qid for qid in a_hits if qid not in b_hits)
    only_b = tuple(qid for qid in b_hits if qid not in a_hits)

    n = len(joined)
    hits_a = sum(1 for qid in joined if a_hits[qid])
    hits_b = sum(1 for qid in joined if b_hits[qid])
    wins = tuple(qid for qid in joined if (not a_hits[qid]) and b_hits[qid])
    losses = tuple(qid for qid in joined if a_hits[qid] and (not b_hits[qid]))

    b_count = len(wins)
    c_count = len(losses)
    delta = (hits_b - hits_a) / n if n > 0 else 0.0
    p_val = mcnemar_exact_p(b_count, c_count)
    ci_a = wilson_ci(hits_a, n)
    ci_b = wilson_ci(hits_b, n)

    if p_val >= 0.05:
        verdict = "inside noise"
    elif b_count > c_count:
        verdict = "significant gain"
    else:
        verdict = "significant regression"

    return PairedComparison(
        metric=metric,
        n=n,
        hits_a=hits_a,
        hits_b=hits_b,
        b_wins=b_count,
        c_losses=c_count,
        delta=delta,
        p_value=p_val,
        ci_a=ci_a,
        ci_b=ci_b,
        flipped_wins=wins,
        flipped_losses=losses,
        only_in_a=only_a,
        only_in_b=only_b,
        verdict=verdict,
    )


def format_comparison(
    res: PairedComparison,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """Format a ``PairedComparison`` as a concise human-readable report."""
    rate_a = res.hits_a / res.n if res.n > 0 else 0.0
    rate_b = res.hits_b / res.n if res.n > 0 else 0.0
    hit_diff = res.hits_b - res.hits_a
    wins_str = ", ".join(res.flipped_wins) if res.flipped_wins else "none"
    losses_str = ", ".join(res.flipped_losses) if res.flipped_losses else "none"

    lines = [
        f"metric: {res.metric}  n: {res.n}",
        (
            f"A ({label_a}): {res.hits_a}/{res.n} ({rate_a:.3f}) "
            f"95% Wilson CI [{res.ci_a[0]:.3f}, {res.ci_a[1]:.3f}]"
        ),
        (
            f"B ({label_b}): {res.hits_b}/{res.n} ({rate_b:.3f}) "
            f"95% Wilson CI [{res.ci_b[0]:.3f}, {res.ci_b[1]:.3f}]"
        ),
        f"delta (B - A): {res.delta:+.3f} ({hit_diff:+d} hits)",
        f"wins/losses (b, c): b={res.b_wins}, c={res.c_losses}",
        f"McNemar exact two-sided p-value: {res.p_value:.6f}",
        f"flipped A=0 -> B=1 ({res.b_wins}): {wins_str}",
        f"flipped A=1 -> B=0 ({res.c_losses}): {losses_str}",
    ]
    if res.only_in_a or res.only_in_b:
        lines.append(
            f"unmatched ids: only_in_a={len(res.only_in_a)}, only_in_b={len(res.only_in_b)}"
        )
    lines.append(f"verdict: {res.verdict}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag_paired_compare.py",
        description="Paired McNemar significance comparison of two --per-query JSONL files.",
    )
    parser.add_argument("a_jsonl", type=Path, help="Baseline per-query JSONL file (arm A).")
    parser.add_argument("b_jsonl", type=Path, help="Candidate per-query JSONL file (arm B).")
    parser.add_argument(
        "--metric",
        choices=ALLOWED_METRICS,
        default="hit4",
        help="Binary hit field to compare (default: hit4).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints paired report and exits 0."""
    args = _parse_args(argv)
    a_hits = load_per_query_hits(args.a_jsonl, args.metric)
    b_hits = load_per_query_hits(args.b_jsonl, args.metric)
    comparison = compare_per_query(a_hits, b_hits, metric=args.metric)
    print(format_comparison(comparison, label_a=str(args.a_jsonl), label_b=str(args.b_jsonl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
