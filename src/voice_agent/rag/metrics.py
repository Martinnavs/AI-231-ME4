"""Offline retrieval metrics, golden-set evaluation, and passage snapshot CLI.

Golden set schema (``config/eval/rag_golden.jsonl``, one JSON object per line)::

    {
        "id": "<unique row id>",
        "q": "<user question>",
        "type": "lexical|numeric|identifier|paraphrase|taglish|unanswerable|off_topic",
        "relevant": [
            {"file": "<corpus filename>", "contains": "<verbatim anchor < 150 chars>"}
        ]
    }

- Answerable types (``lexical``, ``numeric``, ``identifier``, ``paraphrase``, ``taglish``)
  have one or more ``relevant`` anchors inside a single paragraph so chunking never splits
  them. A passage matches when ``passage["source"] == file`` and ``contains`` is a
  substring of ``passage["text"]`` after whitespace normalization.
- Non-answerable types (``unanswerable``, ``off_topic``) have an empty ``relevant`` list
  and are excluded from Recall@k and MRR aggregation.
- Paraphrase questions share at most one content word (per ``doc_chat.corpus.tokenize``)
  with each anchor's ``contains`` text.

Passage snapshot schema (``pipeline/tests/fixtures/rag_passages.json``)::

    {
        "corpus": "tests/fixtures/rag_corpus",
        "k": 4,
        "questions": {
            "<question text>": [
                {
                    "source": "...",
                    "id": "...",
                    "section": "...",
                    "page": null,
                    "file_date": "2026-01-15",
                    "text": "...",
                    "score": 1.2345,
                    "coverage": 0.5
                }
            ]
        }
    }

Generated deterministically over a temporary copy of the corpus with every file mtime set
to ``2026-01-15 00:00 UTC``, ``score`` rounded to 4 decimals, sorted keys, 2-space indent,
and a trailing newline. Questions include every golden ``q`` plus every user-turn
``content`` of cases in ``pipeline/config/eval/benchmark_v1.jsonl`` whose ``id`` does not
start with ``"rag-"``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_agent.rag.docindex import DocIndex
from voice_agent.rag.hybrid import HybridRetriever, make_retriever_factory

NON_ANSWERABLE_TYPES: frozenset[str] = frozenset({"unanswerable", "off_topic"})
SNAPSHOT_MTIME: float = datetime(2026, 1, 15, 0, 0, 0, tzinfo=UTC).timestamp()
DEFAULT_CORPUS_DIR = Path("tests/fixtures/rag_corpus")
DEFAULT_GOLDEN_PATH = Path("config/eval/rag_golden.jsonl")
DEFAULT_BENCHMARK_PATH = Path("pipeline/config/eval/benchmark_v1.jsonl")
DEFAULT_PASSAGES_PATH = Path("pipeline/tests/fixtures/rag_passages.json")
DEFAULT_CACHE_DIR = Path("~/.cache/voice_agent/rag_metrics")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip ends."""
    return " ".join(text.split())


def recall_at_k(flags: Sequence[bool], k: int) -> float:
    """Return 1.0 if any relevant item appears in the top ``k`` flags, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: Sequence[bool]) -> float:
    """Return reciprocal rank (1 / 1-based index) of the first True flag, else 0.0."""
    for idx, hit in enumerate(flags):
        if hit:
            return 1.0 / (idx + 1)
    return 0.0


def is_relevant(
    passage: Mapping[str, Any],
    relevant: Sequence[Mapping[str, Any]],
) -> bool:
    """Return True when ``passage`` matches any ``{file, contains}`` anchor."""
    source = str(passage.get("source") or "")
    text_norm = normalize_whitespace(str(passage.get("text") or ""))
    if not source or not text_norm:
        return False
    for item in relevant:
        file_name = str(item.get("file") or "")
        anchor = normalize_whitespace(str(item.get("contains") or ""))
        if file_name and anchor and source == file_name and anchor in text_norm:
            return True
    return False


def load_golden(path: Path | str) -> list[dict[str, Any]]:
    """Load golden-set rows from a JSONL file."""
    golden_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(
        golden_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"{golden_path}:{line_no}: expected JSON object")
        rows.append(parsed)
    return rows


def load_benchmark_user_turns(path: Path | str) -> list[str]:
    """Return user-turn texts from ``benchmark_v1.jsonl`` cases without a ``rag-`` id prefix."""
    bench_path = Path(path)
    turns_out: list[str] = []
    for raw in bench_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "")
        if case_id.startswith("rag-"):
            continue
        raw_turns = case.get("turns")
        if not isinstance(raw_turns, list):
            continue
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "user":
                content = turn.get("content")
                if isinstance(content, str) and content.strip():
                    turns_out.append(content)
    return turns_out


def _summarize_flags(
    flag_lists: Sequence[list[bool]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    n = len(flag_lists)
    if n == 0:
        return {"n": 0, "recall": {k: 0.0 for k in ks}, "mrr": 0.0}
    return {
        "n": n,
        "recall": {k: sum(recall_at_k(flags, k) for flags in flag_lists) / n for k in ks},
        "mrr": sum(mrr(flags) for flags in flag_lists) / n,
    }


def evaluate(
    search: Callable[[str, int], list[dict[str, Any]]],
    golden: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute overall and per-type Recall@k and MRR on answerable golden rows."""
    ks_tuple = tuple(ks)
    max_k = max(ks_tuple) if ks_tuple else 1

    all_flags: list[list[bool]] = []
    by_type_flags: dict[str, list[list[bool]]] = {}

    for row in golden:
        q_type = str(row.get("type") or "")
        if q_type in NON_ANSWERABLE_TYPES:
            continue
        relevant_raw = row.get("relevant")
        if not isinstance(relevant_raw, Sequence) or isinstance(relevant_raw, (str, bytes)):
            raise ValueError(f"golden row {row.get('id')!r} missing 'relevant' list")
        if not relevant_raw:
            raise ValueError(f"answerable golden row {row.get('id')!r} has empty 'relevant'")

        results = search(str(row["q"]), max_k)
        flags = [is_relevant(passage, relevant_raw) for passage in results[:max_k]]
        all_flags.append(flags)
        by_type_flags.setdefault(q_type, []).append(flags)

    return {
        "overall": _summarize_flags(all_flags, ks_tuple),
        "by_type": {
            q_type: _summarize_flags(flag_lists, ks_tuple)
            for q_type, flag_lists in by_type_flags.items()
        },
    }


def format_metrics_table(report: Mapping[str, Any], ks: Sequence[int] = (1, 3, 5)) -> str:
    """Format an ``evaluate`` result dict as a plain-text table."""
    ks_tuple = tuple(ks)
    headers = ["type", "n", *(f"R@{k}" for k in ks_tuple), "MRR"]

    def _row_cells(label: str, stats: Mapping[str, Any]) -> list[str]:
        recall_map = stats["recall"]
        return [
            label,
            str(stats["n"]),
            *(f"{float(recall_map[k]):.3f}" for k in ks_tuple),
            f"{float(stats['mrr']):.3f}",
        ]

    rows: list[list[str]] = [_row_cells("overall", report["overall"])]
    by_type = report.get("by_type", {})
    for q_type, stats in by_type.items():
        rows.append(_row_cells(str(q_type), stats))

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))
    ]

    def _fmt(cells: Sequence[str]) -> str:
        formatted = [
            cells[0].ljust(widths[0]),
            *(cells[idx].rjust(widths[idx]) for idx in range(1, len(cells))),
        ]
        return " | ".join(formatted)

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt(headers), sep, *(_fmt(row) for row in rows)]
    return "\n".join(lines)


def _corpus_label(folder: Path) -> str:
    if not folder.is_absolute():
        return folder.as_posix()
    try:
        return folder.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return folder.as_posix()


def build_passage_snapshot(
    folder: Path | str = DEFAULT_CORPUS_DIR,
    golden_path: Path | str = DEFAULT_GOLDEN_PATH,
    *,
    k: int = 4,
    benchmark_path: Path | str | None = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    """Build deterministic ``DocIndex.search`` top-``k`` passages for golden + benchmark queries."""
    src_folder = Path(folder)
    golden = load_golden(golden_path)
    questions_list = [str(row["q"]) for row in golden]
    if benchmark_path is not None:
        bench = Path(benchmark_path)
        if bench.exists():
            questions_list.extend(load_benchmark_user_turns(bench))

    unique_questions = list(dict.fromkeys(questions_list))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_corpus = Path(tmp_dir) / "corpus"
        shutil.copytree(
            src_folder,
            tmp_corpus,
            ignore=shutil.ignore_patterns(".*", "~$*"),
        )
        for path in sorted(tmp_corpus.rglob("*")):
            if path.is_file():
                os.utime(path, (SNAPSHOT_MTIME, SNAPSHOT_MTIME))

        index = DocIndex(tmp_corpus)
        index.refresh()

        questions_map: dict[str, list[dict[str, Any]]] = {}
        for query in unique_questions:
            raw_hits = index.search(query, k)
            normalized_hits: list[dict[str, Any]] = []
            for hit in raw_hits:
                copied = dict(hit)
                copied["score"] = round(float(copied["score"]), 4)
                normalized_hits.append(copied)
            questions_map[query] = normalized_hits

    return {
        "corpus": _corpus_label(src_folder),
        "k": k,
        "questions": questions_map,
    }


def dump_passages(
    output_path: Path | str = DEFAULT_PASSAGES_PATH,
    *,
    folder: Path | str = DEFAULT_CORPUS_DIR,
    golden_path: Path | str = DEFAULT_GOLDEN_PATH,
    k: int = 4,
    benchmark_path: Path | str | None = DEFAULT_BENCHMARK_PATH,
) -> Path:
    """Write the deterministic passage snapshot JSON to ``output_path``."""
    payload = build_passage_snapshot(
        folder=folder,
        golden_path=golden_path,
        k=k,
        benchmark_path=benchmark_path,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent.rag.metrics",
        description="Evaluate DocIndex retrieval on the RAG golden set and dump passage snapshots.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Document corpus directory (default: tests/fixtures/rag_corpus).",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Golden evaluation JSONL path (default: config/eval/rag_golden.jsonl).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k passages to capture in --dump-passages (default: 4).",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "hybrid", "auto"),
        default="bm25",
        help="Retrieval backend to evaluate (default: bm25).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="PATH",
        help=(
            "Embedding vector cache directory for --retriever hybrid|auto "
            "(default: ~/.cache/voice_agent/rag_metrics)."
        ),
    )
    parser.add_argument(
        "--dump-passages",
        nargs="?",
        const=str(DEFAULT_PASSAGES_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Write deterministic DocIndex.search snapshot JSON "
            "(default if flag given without path: pipeline/tests/fixtures/rag_passages.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print metrics table and optionally dump the passage snapshot."""
    args = _parse_args(argv)
    golden = load_golden(args.golden)
    retriever: HybridRetriever | None = None
    try:
        if args.retriever == "bm25":
            index = DocIndex(args.folder)
            index.refresh()
            retriever_label = "bm25"
        else:
            cache_dir = Path(args.cache_dir).expanduser()
            factory = make_retriever_factory(
                mode=args.retriever,
                cache_dir=cache_dir,
                background=False,
            )
            created = factory()
            if isinstance(created, HybridRetriever):
                retriever = created
            index = DocIndex(args.folder, retriever_factory=factory)
            index.refresh()
            if retriever is not None:
                if not index.chunks:
                    retriever.build([])
                retriever.wait_ready(timeout=600.0)
                if args.retriever == "hybrid" and not retriever.ready:
                    print(retriever.status(), file=sys.stderr)
                    return 1
                retriever_label = retriever.status()
            else:
                retriever_label = "bm25"

        ks = (1, 3, 5)
        report = evaluate(index.search, golden, ks=ks)
        print(f"retriever: {retriever_label}")
        print(format_metrics_table(report, ks=ks))

        if args.dump_passages is not None:
            out_path = dump_passages(
                args.dump_passages,
                folder=args.folder,
                golden_path=args.golden,
                k=args.k,
            )
            print(f"Wrote passage snapshot to {out_path}")
        return 0
    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    raise SystemExit(main())
