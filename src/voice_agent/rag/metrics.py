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

- Answerable types (``lexical``, ``numeric``, ``identifier``, ``paraphrase``, ``taglish``,
  ``table_cell``, ``spreadsheet_cell``) have one or more ``relevant`` anchors inside a single
  paragraph so chunking never splits them. A passage matches when ``passage["source"] == file``
  and ``contains`` is a substring of ``passage["text"]`` after whitespace normalization
  (``--match exact``) or ``doc_answer.norm``-style folding (``--match normalized``).
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
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_agent.rag import canonical, convert
from voice_agent.rag.backends import ALLOWED_MODES, make_retriever_factory
from voice_agent.rag.docindex import (
    CANONICAL_CONVERTERS,
    DocIndex,
    _resolve_converters,
    decode_and_load,
    load_pdf_pages,
)
from voice_agent.rag.fts5 import Fts5Retriever, Fts5TokenizerMode

NON_ANSWERABLE_TYPES: frozenset[str] = frozenset({"unanswerable", "off_topic"})
MATCH_MODES: tuple[str, ...] = ("exact", "normalized", "tokens")
TOKENS_SPAN_FACTOR = 4
TOKENS_SPAN_SLACK = 40
TOKENS_MAX_PASSAGE_TOKENS = 200_000
TOKENS_MAX_SCAN_STEPS = 2_000_000
TOKENS_MAX_ANCHOR_TOKENS = 200
_UNSUPPORTED_MODE_MSG = "unsupported match mode: {!r} (expected 'exact', 'normalized' or 'tokens')"
GROUP_BY_FIELDS: tuple[str, ...] = ("type", "format", "difficulty")
FTS5_TOKENIZER_MODES: tuple[str, ...] = ("pretok", "porter")
SNAPSHOT_MTIME: float = datetime(2026, 1, 15, 0, 0, 0, tzinfo=UTC).timestamp()
DEFAULT_CORPUS_DIR = Path("tests/fixtures/rag_corpus")
DEFAULT_GOLDEN_PATH = Path("config/eval/rag_golden.jsonl")
DEFAULT_BENCHMARK_PATH = Path("pipeline/config/eval/benchmark_v1.jsonl")
DEFAULT_PASSAGES_PATH = Path("pipeline/tests/fixtures/rag_passages.json")
DEFAULT_CACHE_DIR = Path("~/.cache/voice_agent/rag_metrics")

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_PUNCT_OR_UNDERSCORE_RE = re.compile(r"[^\w' ]|_")

_MARKITDOWN_ARM: dict[str, str] = {
    "md": "identity",
    "txt": "identity",
    "csv": "identity",
    "pdf": "markitdown",
    "docx": "markitdown",
    "pptx": "markitdown",
    "xlsx": "markitdown",
    "html": "markitdown",
    "epub": "markitdown",
}

_PYPDF_ARM: dict[str, str] = {
    "md": "identity",
    "txt": "identity",
    "csv": "identity",
    "pdf": "pypdf",
    "docx": "legacy",
    "pptx": "markitdown",
    "xlsx": "markitdown",
    "html": "markitdown",
    "epub": "markitdown",
}


def resolve_converters_arg(
    converters: str | Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Resolve a converter preset name, ``fmt=conv,...`` spec, or mapping for ``DocIndex``."""
    if converters is None:
        return None
    if isinstance(converters, Mapping):
        return _resolve_converters(converters)

    spec = converters.strip()
    if not spec:
        raise ValueError("converters specification must not be empty")
    lower = spec.lower()
    if lower == "default":
        return None
    if lower == "legacy":
        return {}
    if lower == "markitdown":
        return dict(_MARKITDOWN_ARM)
    if lower == "pypdf":
        return dict(_PYPDF_ARM)

    mapping = dict(CANONICAL_CONVERTERS)
    for raw_pair in spec.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            raise ValueError(
                f"invalid converters spec {converters!r}: expected preset or <fmt=conv,...>"
            )
        fmt_part, mode_part = pair.split("=", 1)
        fmt = fmt_part.strip()
        mode = mode_part.strip()
        if not fmt or not mode:
            raise ValueError(
                f"invalid converters spec {converters!r}: expected preset or <fmt=conv,...>"
            )
        mapping[fmt] = mode
    return _resolve_converters(mapping)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip ends."""
    return " ".join(text.split())


def normalize_for_match(text: str) -> str:
    """Fold case, thousands commas, and punctuation for cross-converter anchor matching."""
    cleaned = _THOUSANDS_RE.sub("", text).lower().replace("’", "'")
    return " ".join(_PUNCT_OR_UNDERSCORE_RE.sub(" ", cleaned).split())


_TOKEN_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize_for_match(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; thousands commas dropped, decimals kept whole."""
    cleaned = _TOKEN_THOUSANDS_RE.sub("", text.lower().replace("’", "'"))
    return _TOKEN_RE.findall(cleaned)


def tokens_match(anchor_tokens: Sequence[str], passage_tokens: Sequence[str]) -> bool:
    """Ordered-subsequence match with a bounded first-to-last span."""
    anchor_tokens = anchor_tokens[:TOKENS_MAX_ANCHOR_TOKENS]
    m = len(anchor_tokens)
    if m == 0:
        return False
    if m == 1 and len(anchor_tokens[0]) < 3 and not any(c.isdigit() for c in anchor_tokens[0]):
        return False
    tokens = list(passage_tokens[:TOKENS_MAX_PASSAGE_TOKENS])
    if not set(anchor_tokens) <= set(tokens):
        return False
    max_span = m * TOKENS_SPAN_FACTOR + TOKENS_SPAN_SLACK
    first = anchor_tokens[0]
    steps = 0
    start = -1
    while True:
        try:
            found = tokens.index(first, start + 1)
        except ValueError:
            return False
        steps += found - start
        start = found
        end = min(len(tokens), start + max_span)
        pos = start + 1
        ok = True
        for tok in anchor_tokens[1:]:
            steps += 1
            try:
                nxt = tokens.index(tok, pos, end) + 1
            except ValueError:
                steps += end - pos
                ok = False
                break
            steps += nxt - pos
            pos = nxt
        if ok:
            return True
        if steps > TOKENS_MAX_SCAN_STEPS:
            return False


def _anchor_in_text(anchor_norm: str, text_norm: str, *, match: str) -> bool:
    if match == "tokens":
        return tokens_match(anchor_norm.split(), text_norm.split())
    return anchor_norm in text_norm


def _normalize_text(text: str, *, match: str) -> str:
    if match == "exact":
        return normalize_whitespace(text)
    if match == "normalized":
        return normalize_for_match(text)
    if match == "tokens":
        return " ".join(tokenize_for_match(text))
    raise ValueError(_UNSUPPORTED_MODE_MSG.format(match))


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
    *,
    match: str = "exact",
) -> bool:
    """Return True when ``passage`` matches any ``{file, contains}`` anchor."""
    if match not in MATCH_MODES:
        raise ValueError(_UNSUPPORTED_MODE_MSG.format(match))
    source = str(passage.get("source") or "")
    text_norm = _normalize_text(str(passage.get("text") or ""), match=match)
    if not source or not text_norm:
        return False
    for item in relevant:
        file_name = str(item.get("file") or "")
        anchor = _normalize_text(str(item.get("contains") or ""), match=match)
        if (
            file_name
            and anchor
            and source == file_name
            and _anchor_in_text(anchor, text_norm, match=match)
        ):
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


def parse_group_by_arg(group_by: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Parse and validate a comma-separated or sequence ``--group-by`` specification."""
    if group_by is None:
        return None
    if isinstance(group_by, str):
        raw_items = [item.strip() for item in group_by.split(",")]
    else:
        raw_items = [str(item).strip() for item in group_by]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError(
            f"invalid --group-by spec {group_by!r}: "
            f"expected comma-separated fields from {', '.join(GROUP_BY_FIELDS)}"
        )
    unknown = [item for item in raw_items if item not in GROUP_BY_FIELDS]
    if unknown:
        raise ValueError(
            f"unknown --group-by field(s): {', '.join(unknown)}; "
            f"expected one of: {', '.join(GROUP_BY_FIELDS)}"
        )
    return tuple(dict.fromkeys(raw_items))


def _resolve_fts5_tokenizer(tokenizer: str) -> Fts5TokenizerMode:
    if tokenizer == "pretok":
        return "pretok"
    if tokenizer == "porter":
        return "porter"
    raise ValueError(f"unsupported fts5 tokenizer: {tokenizer!r} (expected 'pretok' or 'porter')")


def _row_formats(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("format")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    exts: list[str] = []
    for rel in row.get("relevant") or ():
        if isinstance(rel, Mapping):
            fname = str(rel.get("file") or "").strip()
            ext = Path(fname).suffix.lstrip(".").lower()
            if ext:
                exts.append(ext)
    unique = list(dict.fromkeys(exts))
    return unique or ["unknown"]


def _row_difficulties(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("difficulty")
    if isinstance(raw, str):
        val = raw.strip()
        return [val] if val else []
    if isinstance(raw, Sequence) and not isinstance(raw, bytes):
        tags = [str(item).strip() for item in raw if str(item).strip()]
        return list(dict.fromkeys(tags))
    return []


def _row_group_keys(row: Mapping[str, Any], field: str) -> list[str]:
    if field == "type":
        return [str(row.get("type") or "")]
    if field == "format":
        return _row_formats(row)
    if field == "difficulty":
        return _row_difficulties(row)
    raise ValueError(f"unsupported group-by field: {field!r}")


def _build_per_query_row(
    row: Mapping[str, Any],
    q_type: str,
    top_results: Sequence[Mapping[str, Any]],
    relevant_raw: Sequence[Mapping[str, Any]],
    *,
    match: str,
) -> dict[str, Any]:
    """Build one ``--per-query`` JSONL record for an answerable golden row."""
    top_five = list(top_results[:5])
    pq_flags = [is_relevant(passage, relevant_raw, match=match) for passage in top_five]
    first_hit_rank = next((idx + 1 for idx, hit in enumerate(pq_flags) if hit), None)
    return {
        "id": str(row.get("id") or ""),
        "type": q_type,
        "format": _row_formats(row)[0],
        "difficulty": _row_difficulties(row),
        "q": str(row["q"]),
        "hit1": any(pq_flags[:1]),
        "hit3": any(pq_flags[:3]),
        "hit4": any(pq_flags[:4]),
        "hit5": any(pq_flags[:5]),
        "first_hit_rank": first_hit_rank,
        "top": [
            {
                "source": str(passage.get("source") or ""),
                "id": str(passage.get("id") or ""),
                "score": round(float(passage.get("score") or 0.0), 4),
            }
            for passage in top_five
        ],
    }


def evaluate(
    search: Callable[[str, int], list[dict[str, Any]]],
    golden: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = (1, 3, 5),
    *,
    match: str = "exact",
    group_by: str | Sequence[str] | None = None,
    per_query_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute overall and per-group Recall@k and MRR on answerable golden rows."""
    if match not in MATCH_MODES:
        raise ValueError(_UNSUPPORTED_MODE_MSG.format(match))
    parsed_groups = parse_group_by_arg(group_by)
    active_groups = parsed_groups if parsed_groups is not None else ("type",)
    ks_tuple = tuple(ks)
    max_k = max(ks_tuple) if ks_tuple else 1
    search_k = max(max_k, 5) if per_query_out is not None else max_k

    all_flags: list[list[bool]] = []
    grouped_flags: dict[str, dict[str, list[list[bool]]]] = {field: {} for field in active_groups}

    for row in golden:
        q_type = str(row.get("type") or "")
        if q_type in NON_ANSWERABLE_TYPES:
            continue
        relevant_raw = row.get("relevant")
        if not isinstance(relevant_raw, Sequence) or isinstance(relevant_raw, (str, bytes)):
            raise ValueError(f"golden row {row.get('id')!r} missing 'relevant' list")
        if not relevant_raw:
            raise ValueError(f"answerable golden row {row.get('id')!r} has empty 'relevant'")

        results = search(str(row["q"]), search_k)
        flags = [is_relevant(passage, relevant_raw, match=match) for passage in results[:max_k]]
        all_flags.append(flags)
        if per_query_out is not None:
            per_query_out.append(
                _build_per_query_row(row, q_type, results, relevant_raw, match=match)
            )
        for field in active_groups:
            for key in _row_group_keys(row, field):
                grouped_flags[field].setdefault(key, []).append(flags)

    result: dict[str, Any] = {"overall": _summarize_flags(all_flags, ks_tuple)}
    for field in active_groups:
        result[f"by_{field}"] = {
            key: _summarize_flags(flag_lists, ks_tuple)
            for key, flag_lists in grouped_flags[field].items()
        }
    return result


def _format_single_group_table(
    label_col: str,
    overall: Mapping[str, Any],
    groups: Mapping[str, Any],
    ks_tuple: tuple[int, ...],
) -> str:
    headers = [label_col, "n", *(f"R@{k}" for k in ks_tuple), "MRR"]

    def _recall_val(recall_map: Mapping[Any, Any], k: int) -> float:
        if k in recall_map:
            return float(recall_map[k])
        return float(recall_map.get(str(k), 0.0))

    def _row_cells(label: str, stats: Mapping[str, Any]) -> list[str]:
        recall_map = stats["recall"]
        return [
            label,
            str(stats["n"]),
            *(f"{_recall_val(recall_map, k):.3f}" for k in ks_tuple),
            f"{float(stats['mrr']):.3f}",
        ]

    rows: list[list[str]] = [_row_cells("overall", overall)]
    for key, stats in groups.items():
        rows.append(_row_cells(str(key), stats))

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


def format_metrics_table(report: Mapping[str, Any], ks: Sequence[int] = (1, 3, 5)) -> str:
    """Format an ``evaluate`` result dict as one or more plain-text tables."""
    ks_tuple = tuple(ks)
    present_fields = [field for field in GROUP_BY_FIELDS if f"by_{field}" in report]
    if not present_fields:
        return _format_single_group_table(
            "type", report["overall"], report.get("by_type", {}), ks_tuple
        )
    return "\n\n".join(
        _format_single_group_table(field, report["overall"], report[f"by_{field}"], ks_tuple)
        for field in present_fields
    )


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
    converters: str | Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build deterministic ``DocIndex.search`` top-``k`` passages for golden + benchmark queries."""
    resolved_converters = resolve_converters_arg(converters)
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
        base_ignore = shutil.ignore_patterns(".*", "~$*")

        def _ignore_symlinks_and_hidden(dirpath: str, names: list[str]) -> set[str]:
            ignored = set(base_ignore(dirpath, names))
            parent = Path(dirpath)
            for name in names:
                if (parent / name).is_symlink():
                    ignored.add(name)
            return ignored

        shutil.copytree(
            src_folder,
            tmp_corpus,
            symlinks=True,
            ignore=_ignore_symlinks_and_hidden,
        )
        for path in sorted(tmp_corpus.rglob("*")):
            if not path.is_symlink() and path.is_file():
                os.utime(path, (SNAPSHOT_MTIME, SNAPSHOT_MTIME))

        index = DocIndex(
            tmp_corpus,
            converters=resolved_converters,
            canonical_cache_dir=None,
        )
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
    converters: str | Mapping[str, str] | None = None,
) -> Path:
    """Write the deterministic passage snapshot JSON to ``output_path``."""
    payload = build_passage_snapshot(
        folder=folder,
        golden_path=golden_path,
        k=k,
        benchmark_path=benchmark_path,
        converters=converters,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def parse_compare_arms_arg(compare_arms: str) -> list[str]:
    """Split and validate a comma-separated ``--compare-arms`` specification."""
    raw_items = [item.strip() for item in compare_arms.split(",")]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError(
            f"invalid --compare-arms spec {compare_arms!r}: expected comma-separated arm names"
        )
    for item in raw_items:
        resolve_converters_arg(item)
    return raw_items


def _peak_rss_bytes() -> tuple[int, int, int]:
    """Return ``(peak_bytes, self_bytes, children_bytes)`` from ``resource.getrusage``."""
    try:
        import resource
    except ImportError:
        return 0, 0, 0
    unit = 1 if sys.platform == "darwin" else 1024
    self_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * unit
    child_bytes = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * unit
    return max(self_bytes, child_bytes), self_bytes, child_bytes


def _extract_converted_locator_lines(raw_text: str, *, kind: str) -> list[str]:
    """Return raw pptx/xlsx locator lines that ``postprocess_markdown`` turns into markers."""
    if kind not in ("pptx", "xlsx"):
        return []
    converted: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    last_slide = 0
    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        next_char, next_len = canonical._fence_transition(line, fence_char, fence_len)
        if fence_char is not None or next_char is not None:
            fence_char, fence_len = next_char, next_len
            continue
        if kind == "pptx":
            slide_num = canonical._parse_pptx_slide_number(line, expected=last_slide + 1)
            if slide_num is not None:
                converted.append(line)
                last_slide = slide_num
        elif kind == "xlsx" and line.startswith("## ") and canonical.clean_sheet_name(line[3:]):
            converted.append(line)
    return converted


def _convert_raw_and_canonical(
    raw_bytes: bytes,
    kind: str,
    mode: str,
) -> tuple[str, str, list[str], list[str] | None]:
    """Return ``(raw_text, canonical_text, dropped_lines, legacy_pdf_pages)`` for diagnostics."""
    if mode == "identity":
        loaded, status, _ = decode_and_load(raw_bytes, kind)
        if status != "ok":
            return "", "", [], None
        return loaded, canonical.neutralize_markers(loaded), [], None

    if mode == "legacy":
        if kind == "pdf":
            pages, status, _ = load_pdf_pages(raw_bytes)
            if status != "ok":
                return "", "", [], None
            raw_pages = [text for _, text in pages]
            joined = "\n\n".join(raw_pages)
            return joined, joined, [], raw_pages
        loaded, status, _ = decode_and_load(raw_bytes, kind)
        if status != "ok":
            return "", "", [], None
        return loaded, loaded, [], None

    if mode == "pypdf":
        res = convert.convert_pdf(raw_bytes, mode="pypdf")
        if res.status != "ok":
            return "", "", [], None
        pages, _, _ = load_pdf_pages(raw_bytes)
        slot_count = max((p for p, _ in pages), default=0)
        page_list = [""] * slot_count
        for p_num, p_text in pages:
            page_list[p_num - 1] = p_text
        pp = canonical.postprocess_pdf_pages(page_list)
        return res.raw_text, res.text, pp.dropped_lines, None

    if mode == "markitdown":
        res = convert.convert_with_markitdown(raw_bytes, kind)
        if res.status != "ok":
            return "", "", [], None
        if kind == "pdf":
            if res.fallback is None and "\f" in res.raw_text:
                page_count, _ = convert._pdf_metadata(raw_bytes, [])
                md_pages = convert._split_pdf_pages(res.raw_text, page_count)
                if md_pages is not None:
                    pp = canonical.postprocess_pdf_pages(md_pages)
                    return res.raw_text, res.text, pp.dropped_lines, None
            pages, _, _ = load_pdf_pages(raw_bytes)
            slot_count = max((p for p, _ in pages), default=0)
            page_list = [""] * slot_count
            for p_num, p_text in pages:
                page_list[p_num - 1] = p_text
            pp = canonical.postprocess_pdf_pages(page_list)
            return res.raw_text, res.text, pp.dropped_lines, None
        locator_lines = _extract_converted_locator_lines(res.raw_text, kind=kind)
        return res.raw_text, res.text, locator_lines, None

    if mode == "direct":
        res = convert.convert_with_direct(raw_bytes, kind)
        if res.status != "ok":
            return "", "", [], None
        if res.raw_text:
            locator_lines = _extract_converted_locator_lines(res.raw_text, kind=kind)
            return res.raw_text, res.text, locator_lines, None
        # Direct readers normally populate raw_text via _finalize_markdown; if absent,
        # use the extracted plain text (markers stripped) so digit counts stay aligned.
        return canonical.strip_markers(res.text), res.text, [], None

    raise ValueError(f"unexpected converter mode: {mode!r}")


def _pages_ending_mid_sentence_count(
    canon_text: str,
    legacy_pdf_pages: Sequence[str] | None,
) -> int:
    """Count non-final PDF pages whose last non-empty line lacks terminal punctuation."""
    bodies: list[str]
    if legacy_pdf_pages is not None:
        bodies = list(legacy_pdf_pages)
    else:
        units, _ = canonical.split_units(canon_text)
        bodies = [unit.body for unit in units]

    count = 0
    for body in bodies[:-1]:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines and not lines[-1].endswith((".", "!", "?", ":")):
            count += 1
    return count


def _build_index_for_eval(
    folder: Path,
    *,
    retriever_mode: str,
    cache_dir: Path | None,
    converters: Mapping[str, str] | None,
    fts5_tokenizer: str = "pretok",
) -> tuple[DocIndex, Fts5Retriever | None, str]:
    """Construct and refresh a ``DocIndex`` with ``canonical_cache_dir=None``."""
    if retriever_mode == "bm25":
        index = DocIndex(
            folder,
            converters=converters,
            canonical_cache_dir=None,
        )
        index.refresh()
        return index, None, "bm25"

    if retriever_mode == "fts5":
        tokenizer_lit = _resolve_fts5_tokenizer(fts5_tokenizer)
        expanded_cache = Path(cache_dir).expanduser() if cache_dir is not None else None
        factory = make_retriever_factory(
            mode="fts5",
            cache_dir=expanded_cache,
            fts5_tokenizer=tokenizer_lit,
        )
        created = factory()
        fts5_retriever = created if isinstance(created, Fts5Retriever) else None
        if fts5_retriever is not None:
            status_msg = fts5_retriever.status()
            if status_msg.startswith("bm25 (fts5 unavailable:"):
                fts5_retriever.close()
                raise RuntimeError(status_msg)
        try:
            index = DocIndex(
                folder,
                retriever_factory=factory,
                converters=converters,
                canonical_cache_dir=None,
            )
            index.refresh()
            if fts5_retriever is not None:
                status_msg = fts5_retriever.status()
                if status_msg.startswith("bm25 (fts5 unavailable:"):
                    raise RuntimeError(status_msg)
                return index, fts5_retriever, status_msg
            return index, None, "fts5"
        except Exception:
            if fts5_retriever is not None:
                fts5_retriever.close()
            raise

    if retriever_mode != "auto":
        raise ValueError(f"unsupported retriever mode: {retriever_mode!r}")

    expanded_cache = Path(DEFAULT_CACHE_DIR if cache_dir is None else cache_dir).expanduser()
    factory = make_retriever_factory(mode="auto", cache_dir=expanded_cache)
    created = factory()
    fts5_retriever = created if isinstance(created, Fts5Retriever) else None
    try:
        index = DocIndex(
            folder,
            retriever_factory=factory,
            converters=converters,
            canonical_cache_dir=None,
        )
        index.refresh()
        if fts5_retriever is not None:
            return index, fts5_retriever, fts5_retriever.status()
        return index, None, "bm25"
    except Exception:
        if fts5_retriever is not None:
            fts5_retriever.close()
        raise


def evaluate_arm(
    arm_spec: str,
    *,
    folder: Path | str,
    golden_path: Path | str,
    retriever_mode: str = "bm25",
    cache_dir: Path | str | None = None,
    match: str = "exact",
    ks: Sequence[int] = (1, 3, 5),
    fts5_tokenizer: str = "pretok",
    group_by: str | Sequence[str] | None = None,
    per_query_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate one converter arm over ``folder`` and ``golden_path`` without folder writes."""
    if match not in MATCH_MODES:
        raise ValueError(_UNSUPPORTED_MODE_MSG.format(match))
    parsed_groups = parse_group_by_arg(group_by)
    tokenizer_lit = _resolve_fts5_tokenizer(fts5_tokenizer)
    folder_path = Path(folder)
    resolved_converters = resolve_converters_arg(arm_spec)
    effective_converters = _resolve_converters(resolved_converters)
    golden = load_golden(golden_path)

    retriever: Fts5Retriever | None = None
    try:
        t_refresh0 = time.perf_counter()
        index, retriever, retriever_label = _build_index_for_eval(
            folder_path,
            retriever_mode=retriever_mode,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            converters=resolved_converters,
            fts5_tokenizer=tokenizer_lit,
        )
        refresh_s = time.perf_counter() - t_refresh0
        metrics_report = evaluate(
            index.search,
            golden,
            ks=ks,
            match=match,
            group_by=parsed_groups,
            per_query_out=per_query_out,
        )

        chunks_by_source: dict[str, list[Any]] = {}
        chunks_norm_by_source: dict[str, list[str]] = {}
        for chunk in index.chunks:
            chunks_by_source.setdefault(chunk.source, []).append(chunk)
            chunks_norm_by_source.setdefault(chunk.source, []).append(
                _normalize_text(chunk.text, match=match)
            )

        file_entries: list[dict[str, Any]] = []
        joined_norm_by_source: dict[str, str] = {}
        total_conversion_s = 0.0
        pages_ending_mid_sentence = 0
        mismatched_files: list[str] = []
        checked_digit_files = 0

        for rec in index.records():
            file_chunks = chunks_by_source.get(rec.name, [])
            converters_used = sorted({c.converter for c in file_chunks})
            fallback: str | None = None
            if any("fallback=pypdf-pages" in conv for conv in converters_used):
                fallback = "pypdf-pages"
            elif any("fallback=legacy" in conv for conv in converters_used):
                fallback = "legacy"

            if rec.status != "ok":
                file_entries.append(
                    {
                        "name": rec.name,
                        "kind": rec.kind,
                        "status": rec.status,
                        "error": rec.error,
                        "converters": converters_used,
                        "fallback": fallback,
                        "chunks": len(file_chunks),
                        "conversion_s": 0.0,
                        "digit_mismatch": False,
                        "digit_check": None,
                    }
                )
                continue

            mode = effective_converters.get(rec.kind, "legacy")
            conv_dt = 0.0
            digit_check: dict[str, Any] | None = None
            digit_mismatch = False
            canon_text = ""
            legacy_pdf_pages: list[str] | None = None

            try:
                raw_bytes = Path(rec.path).read_bytes()
                t_conv0 = time.perf_counter()
                raw_text, canon_text, dropped_lines, legacy_pdf_pages = _convert_raw_and_canonical(
                    raw_bytes,
                    rec.kind,
                    mode,
                )
                conv_dt = time.perf_counter() - t_conv0
                total_conversion_s += conv_dt

                if canon_text:
                    raw_runs = Counter(canonical.digit_runs(raw_text))
                    canon_runs = Counter(canonical.digit_runs(canonical.strip_markers(canon_text)))
                    dropped_runs = Counter(canonical.digit_runs("\n".join(dropped_lines)))
                    actual_runs = canon_runs + dropped_runs
                    missing = sorted((raw_runs - actual_runs).elements())
                    extra = sorted((actual_runs - raw_runs).elements())
                    digit_mismatch = actual_runs != raw_runs
                    digit_check = {
                        "mismatch": digit_mismatch,
                        "raw_count": sum(raw_runs.values()),
                        "canonical_count": sum(canon_runs.values()),
                        "dropped_count": sum(dropped_runs.values()),
                        "missing": missing,
                        "extra": extra,
                    }
                    checked_digit_files += 1
                    if digit_mismatch:
                        mismatched_files.append(rec.name)
            except Exception as exc:
                digit_check = {
                    "mismatch": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            joined_source_text = (
                canonical.strip_markers(canon_text)
                if canon_text
                else "\n".join(c.text for c in file_chunks)
            )
            joined_norm_by_source[rec.name] = _normalize_text(joined_source_text, match=match)

            if rec.kind == "pdf":
                pages_ending_mid_sentence += _pages_ending_mid_sentence_count(
                    canon_text,
                    legacy_pdf_pages,
                )

            file_entries.append(
                {
                    "name": rec.name,
                    "kind": rec.kind,
                    "status": rec.status,
                    "error": rec.error,
                    "converters": converters_used,
                    "fallback": fallback,
                    "chunks": len(file_chunks),
                    "conversion_s": round(conv_dt, 6),
                    "digit_mismatch": digit_mismatch,
                    "digit_check": digit_check,
                }
            )

        split_details: list[dict[str, str]] = []
        for row in golden:
            if str(row.get("type") or "") in NON_ANSWERABLE_TYPES:
                continue
            row_id = str(row.get("id") or "")
            for rel in row.get("relevant", []):
                file_name = str(rel.get("file") or "")
                anchor = str(rel.get("contains") or "")
                anchor_norm = _normalize_text(anchor, match=match)
                if not file_name or not anchor_norm:
                    continue
                in_joined = _anchor_in_text(
                    anchor_norm, joined_norm_by_source.get(file_name, ""), match=match
                )
                in_single_chunk = any(
                    _anchor_in_text(anchor_norm, c_text, match=match)
                    for c_text in chunks_norm_by_source.get(file_name, [])
                )
                if in_joined and not in_single_chunk:
                    split_details.append({"id": row_id, "file": file_name, "contains": anchor})

        ok_files = [entry for entry in file_entries if entry["status"] == "ok"]
        pdf_files = [entry for entry in file_entries if entry["kind"] == "pdf"]
        ok_pdfs = [entry for entry in pdf_files if entry["status"] == "ok"]
        pdf_pypdf_fallback_count = sum(1 for entry in ok_pdfs if entry["fallback"] == "pypdf-pages")
        legacy_fallback_count = sum(1 for entry in ok_files if entry["fallback"] == "legacy")

        peak_rss_bytes, self_rss_bytes, children_rss_bytes = _peak_rss_bytes()

        return {
            "arm": arm_spec,
            "converters": effective_converters,
            "retriever": retriever_label,
            "match": match,
            "metrics": metrics_report,
            "timings": {
                "conversion_s": round(total_conversion_s, 6),
                "refresh_s": round(refresh_s, 6),
            },
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mb": round(peak_rss_bytes / (1024 * 1024), 2),
            "peak_rss_self_bytes": self_rss_bytes,
            "peak_rss_children_bytes": children_rss_bytes,
            "files": file_entries,
            "fallback_rates": {
                "pdf_pypdf_fallback_count": pdf_pypdf_fallback_count,
                "pdf_ok_count": len(ok_pdfs),
                "pdf_total_count": len(pdf_files),
                "pdf_pypdf_fallback_rate": (
                    pdf_pypdf_fallback_count / len(ok_pdfs) if ok_pdfs else 0.0
                ),
                "legacy_fallback_count": legacy_fallback_count,
                "ok_file_count": len(ok_files),
                "total_file_count": len(file_entries),
                "legacy_fallback_rate": (
                    legacy_fallback_count / len(ok_files) if ok_files else 0.0
                ),
            },
            "cross_page_splits": {
                "count": len(split_details),
                "pages_ending_mid_sentence": pages_ending_mid_sentence,
                "details": split_details,
            },
            "digit_mismatches": {
                "mismatch_count": len(mismatched_files),
                "checked_files": checked_digit_files,
                "mismatched_files": mismatched_files,
            },
        }
    finally:
        if retriever is not None:
            retriever.close()


def _run_arm_subprocess(
    arm_spec: str,
    *,
    folder: Path,
    golden_path: Path,
    retriever_mode: str,
    cache_dir: Path | None,
    match: str,
    fts5_tokenizer: Fts5TokenizerMode = "pretok",
    group_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute ``evaluate_arm`` in a dedicated subprocess so peak RSS is isolated per arm."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        arm_out = Path(tmp_dir) / "arm_report.json"
        cmd = [
            sys.executable,
            "-m",
            "voice_agent.rag.metrics",
            "--folder",
            str(folder),
            "--golden",
            str(golden_path),
            "--retriever",
            retriever_mode,
            "--converters",
            arm_spec,
            "--match",
            match,
            "--_arm-output",
            str(arm_out),
        ]
        if cache_dir is not None:
            cmd.extend(["--cache-dir", str(cache_dir)])
        if retriever_mode == "fts5":
            cmd.extend(["--fts5-tokenizer", fts5_tokenizer])
        if group_by is not None:
            cmd.extend(["--group-by", ",".join(group_by)])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err_detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RuntimeError(f"arm {arm_spec!r} failed: {err_detail}")
        if not arm_out.is_file():
            raise RuntimeError(f"arm {arm_spec!r} produced no report file")
        parsed = json.loads(arm_out.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"arm {arm_spec!r} returned non-object report")
        return parsed


def compare_arms(
    arm_specs: Sequence[str],
    *,
    folder: Path | str,
    golden_path: Path | str,
    retriever_mode: str = "bm25",
    cache_dir: Path | str | None = None,
    match: str = "exact",
    fts5_tokenizer: str = "pretok",
    group_by: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run each arm in its own subprocess over ``folder``/``golden_path`` and aggregate results."""
    if match not in MATCH_MODES:
        raise ValueError(_UNSUPPORTED_MODE_MSG.format(match))
    parsed_groups = parse_group_by_arg(group_by)
    tokenizer_lit = _resolve_fts5_tokenizer(fts5_tokenizer)
    validated_arms = [item.strip() for item in arm_specs]
    if not validated_arms or any(not item for item in validated_arms):
        raise ValueError("compare_arms requires at least one non-empty arm specification")
    for arm in validated_arms:
        resolve_converters_arg(arm)

    folder_path = Path(folder)
    golden = Path(golden_path)
    cache = Path(cache_dir) if cache_dir is not None else None

    arm_reports = [
        _run_arm_subprocess(
            arm,
            folder=folder_path,
            golden_path=golden,
            retriever_mode=retriever_mode,
            cache_dir=cache,
            match=match,
            fts5_tokenizer=tokenizer_lit,
            group_by=parsed_groups,
        )
        for arm in validated_arms
    ]
    return {
        "folder": str(folder_path),
        "golden": str(golden),
        "retriever": retriever_mode,
        "match": match,
        "arms": arm_reports,
    }


def format_arm_report(arm_report: Mapping[str, Any], ks: Sequence[int] = (1, 3, 5)) -> str:
    """Format one arm's metrics table and diagnostic summary for stdout."""
    lines = [
        (
            f"=== arm: {arm_report['arm']} "
            f"(retriever: {arm_report['retriever']}, match: {arm_report['match']}) ==="
        ),
        format_metrics_table(arm_report["metrics"], ks=ks),
    ]
    timings = arm_report["timings"]
    fb = arm_report["fallback_rates"]
    splits = arm_report["cross_page_splits"]
    digits = arm_report["digit_mismatches"]
    files = arm_report["files"]
    ok_count = fb["ok_file_count"]
    total_count = fb["total_file_count"]
    lines.append(
        f"timings: conversion={timings['conversion_s']:.4f}s, "
        f"refresh={timings['refresh_s']:.4f}s, "
        f"peak_rss={arm_report['peak_rss_mb']:.2f} MB"
    )
    lines.append(
        f"fallbacks: pdf_pypdf={fb['pdf_pypdf_fallback_count']}/{fb['pdf_ok_count']} "
        f"({fb['pdf_pypdf_fallback_rate'] * 100:.1f}%), "
        f"legacy={fb['legacy_fallback_count']}/{ok_count} "
        f"({fb['legacy_fallback_rate'] * 100:.1f}%)"
    )
    lines.append(
        f"cross_page_splits: {splits['count']} "
        f"(pages_ending_mid_sentence={splits['pages_ending_mid_sentence']})"
    )
    lines.append(f"digit_mismatches: {digits['mismatch_count']}/{digits['checked_files']} files")
    lines.append(f"files: {ok_count}/{total_count} ok")
    for entry in files:
        conv_str = ",".join(entry["converters"]) if entry["converters"] else "-"
        parts = [
            f"  - {entry['name']}: status={entry['status']}",
            f"kind={entry['kind']}",
            f"converter={conv_str}",
        ]
        if entry["fallback"]:
            parts.append(f"fallback={entry['fallback']}")
        if entry["digit_mismatch"]:
            parts.append("digit_mismatch=true")
        if entry["error"]:
            parts.append(f"error={entry['error']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


class _DumpPassagesAction(argparse.Action):
    """Record ``--dump-passages`` target path and whether a path was explicitly provided."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        explicit = values is not None
        target = str(values) if explicit else str(DEFAULT_PASSAGES_PATH)
        setattr(namespace, self.dest, target)
        namespace._dump_passages_explicit = explicit


class _CacheDirAction(argparse.Action):
    """Record ``--cache-dir`` path and whether it was explicitly supplied."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if values is None:
            return
        target = values if isinstance(values, Path) else Path(str(values))
        setattr(namespace, self.dest, target)
        namespace._cache_dir_explicit = True


def _is_default_folder(folder: Path) -> bool:
    try:
        return folder.resolve() == DEFAULT_CORPUS_DIR.resolve()
    except OSError:
        return folder == DEFAULT_CORPUS_DIR


def _write_report_json(path: Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _is_path_inside_folder(path: Path, folder: Path) -> bool:
    """Return True if ``path`` resolves or lexically sits inside ``folder``."""
    abs_path = Path(os.path.abspath(path))
    abs_folder = Path(os.path.abspath(folder))
    if abs_path == abs_folder or abs_path.is_relative_to(abs_folder):
        return True
    try:
        resolved_path = path.resolve()
        resolved_folder = folder.resolve()
    except OSError:
        return False
    return resolved_path == resolved_folder or resolved_path.is_relative_to(resolved_folder)


def _write_per_query_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    folder: Path | None = None,
) -> Path:
    out = Path(path)
    if folder is not None and _is_path_inside_folder(out, folder):
        raise ValueError(f"--per-query PATH ({out}) must not be inside --folder ({folder})")
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    out.write_text(content, encoding="utf-8")
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
        choices=ALLOWED_MODES,
        default="bm25",
        help="Retrieval backend to evaluate (default: bm25).",
    )
    parser.add_argument(
        "--fts5-tokenizer",
        choices=FTS5_TOKENIZER_MODES,
        default="pretok",
        help="Tokenizer mode when --retriever fts5 (default: pretok).",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        metavar="FIELDS",
        help="Comma-separated grouping fields: type,format,difficulty (default: type).",
    )
    parser.add_argument(
        "--converters",
        default="default",
        help=(
            "Per-format converter preset or overrides: "
            "legacy|default|markitdown|pypdf|<fmt=conv,...> (default: default)."
        ),
    )
    parser.add_argument(
        "--compare-arms",
        default=None,
        metavar="ARM,...",
        help="Comma-separated converter arms to evaluate in separate subprocesses.",
    )
    parser.add_argument(
        "--match",
        choices=MATCH_MODES,
        default="exact",
        help=(
            "Anchor matching mode: exact (whitespace), normalized (doc_answer.norm style) "
            "or tokens (converter-independent ordered tokens)."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full evaluation and per-file diagnostic report JSON to PATH.",
    )
    parser.add_argument(
        "--per-query",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write per-query JSONL rows for answerable golden queries to PATH "
            "(must not be inside --folder)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        action=_CacheDirAction,
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="PATH",
        help=(
            "Cache directory for --retriever auto|fts5 "
            "(default: ~/.cache/voice_agent/rag_metrics; fts5 uses in-memory DB unless set)."
        ),
    )
    parser.add_argument(
        "--dump-passages",
        action=_DumpPassagesAction,
        nargs="?",
        const=None,
        default=None,
        metavar="PATH",
        help=(
            "Write deterministic DocIndex.search snapshot JSON "
            "(default if flag given without path: pipeline/tests/fixtures/rag_passages.json; "
            "explicit PATH required when --folder is not the default)."
        ),
    )
    parser.add_argument(
        "--_arm-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(_dump_passages_explicit=False, _cache_dir_explicit=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print metrics table and optionally dump the passage snapshot."""
    args = _parse_args(argv)

    try:
        group_by_fields = parse_group_by_arg(args.group_by)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    effective_cache_dir: Path | None = (
        args.cache_dir
        if (args.retriever != "fts5" or getattr(args, "_cache_dir_explicit", False))
        else None
    )

    if args.dump_passages is not None:
        if not str(args.dump_passages).strip():
            print("error: --dump-passages PATH must not be empty", file=sys.stderr)
            return 2
        if not getattr(args, "_dump_passages_explicit", False) and not _is_default_folder(
            args.folder
        ):
            print(
                "error: --dump-passages without an explicit PATH is refused "
                "when --folder is not the default",
                file=sys.stderr,
            )
            return 2
        if args.compare_arms is not None:
            print("error: --dump-passages cannot be combined with --compare-arms", file=sys.stderr)
            return 2

    if args.per_query is not None:
        if not str(args.per_query).strip():
            print("error: --per-query PATH must not be empty", file=sys.stderr)
            return 2
        if args.compare_arms is not None:
            print("error: --per-query cannot be combined with --compare-arms", file=sys.stderr)
            return 2
        if _is_path_inside_folder(args.per_query, args.folder):
            print(
                f"error: --per-query PATH ({args.per_query}) must not be inside "
                f"--folder ({args.folder})",
                file=sys.stderr,
            )
            return 2

    if args._arm_output is not None:
        try:
            arm_report = evaluate_arm(
                args.converters,
                folder=args.folder,
                golden_path=args.golden,
                retriever_mode=args.retriever,
                cache_dir=effective_cache_dir,
                match=args.match,
                fts5_tokenizer=args.fts5_tokenizer,
                group_by=group_by_fields,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _write_report_json(args._arm_output, arm_report)
        return 0

    if args.compare_arms is not None:
        try:
            arm_specs = parse_compare_arms_arg(args.compare_arms)
            comparison = compare_arms(
                arm_specs,
                folder=args.folder,
                golden_path=args.golden,
                retriever_mode=args.retriever,
                cache_dir=effective_cache_dir,
                match=args.match,
                fts5_tokenizer=args.fts5_tokenizer,
                group_by=group_by_fields,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for idx, arm_report in enumerate(comparison["arms"]):
            if idx > 0:
                print()
            print(format_arm_report(arm_report))
        if args.report_json is not None:
            out_report = _write_report_json(args.report_json, comparison)
            print(f"Wrote report JSON to {out_report}")
        return 0

    try:
        resolved_converters = resolve_converters_arg(args.converters)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.report_json is not None:
        per_query_rows: list[dict[str, Any]] | None = [] if args.per_query is not None else None
        try:
            arm_report = evaluate_arm(
                args.converters,
                folder=args.folder,
                golden_path=args.golden,
                retriever_mode=args.retriever,
                cache_dir=effective_cache_dir,
                match=args.match,
                fts5_tokenizer=args.fts5_tokenizer,
                group_by=group_by_fields,
                per_query_out=per_query_rows,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"retriever: {arm_report['retriever']}")
        print(format_metrics_table(arm_report["metrics"]))
        payload = {
            "folder": str(args.folder),
            "golden": str(args.golden),
            "retriever": args.retriever,
            "match": args.match,
            "arms": [arm_report],
        }
        out_report = _write_report_json(args.report_json, payload)
        print(f"Wrote report JSON to {out_report}")
        if args.per_query is not None and per_query_rows is not None:
            out_pq = _write_per_query_jsonl(args.per_query, per_query_rows, folder=args.folder)
            print(f"Wrote per-query JSONL to {out_pq}")
        if args.dump_passages is not None:
            out_path = dump_passages(
                args.dump_passages,
                folder=args.folder,
                golden_path=args.golden,
                k=args.k,
                converters=resolved_converters,
            )
            print(f"Wrote passage snapshot to {out_path}")
        return 0

    golden = load_golden(args.golden)
    retriever: Fts5Retriever | None = None
    try:
        index, retriever, retriever_label = _build_index_for_eval(
            args.folder,
            retriever_mode=args.retriever,
            cache_dir=effective_cache_dir,
            converters=resolved_converters,
            fts5_tokenizer=args.fts5_tokenizer,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        ks = (1, 3, 5)
        per_query_rows = [] if args.per_query is not None else None
        report = evaluate(
            index.search,
            golden,
            ks=ks,
            match=args.match,
            group_by=group_by_fields,
            per_query_out=per_query_rows,
        )
        print(f"retriever: {retriever_label}")
        print(format_metrics_table(report, ks=ks))

        if args.per_query is not None and per_query_rows is not None:
            out_pq = _write_per_query_jsonl(args.per_query, per_query_rows, folder=args.folder)
            print(f"Wrote per-query JSONL to {out_pq}")

        if args.dump_passages is not None:
            out_path = dump_passages(
                args.dump_passages,
                folder=args.folder,
                golden_path=args.golden,
                k=args.k,
                converters=resolved_converters,
            )
            print(f"Wrote passage snapshot to {out_path}")
        return 0
    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    raise SystemExit(main())
