"""Validate an implicit-compound question set for the sufficiency experiment.

The validator is corpus-agnostic: the set file and the corpus root are given
on the command line, and no corpus-specific text appears here. It checks the
set's structure (row count, kind balance, ids, required keys, question style)
and verifies that each row's ``evidence`` snippet occurs in the extracted text
of the corpus file it cites.

Usage:
    python scripts/check_implicit_set.py \
        --set eval/sufficiency/implicit_compound.jsonl \
        --root /path/to/corpus

Add ``--skip-evidence`` to skip the evidence-in-file check (e.g. when no
parser is available for the corpus formats).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

KINDS = (
    "needs_web_stale",
    "needs_web_missing_half",
    "needs_web_conflict",
    "docs_only_control",
)
EXPECTED_KIND_COUNTS: dict[str, int] = {
    "needs_web_stale": 20,
    "needs_web_missing_half": 15,
    "needs_web_conflict": 10,
    "docs_only_control": 15,
}
TOTAL_ROWS = 60
MIN_DISTINCT_FILES = 25
MIN_QUESTION_WORDS = 6
MAX_QUESTION_WORDS = 22
MAX_EVIDENCE_CHARS = 200
REQUIRED_KEYS = (
    "id",
    "kind",
    "question",
    "doc_fact",
    "doc_file",
    "web_needed",
    "web_fact",
    "why",
    "ideal_answer",
    "stale_marker",
    "evidence",
    "split",
)
ID_RE = re.compile(r"^[a-z][a-z0-9]*-\d{3}$")
# A question must read as an ordinary spoken question: it may not name the
# sources (corpus documents or the web) in any form.
FORBIDDEN_QUESTION_WORDS = ("document", "file", "note", "web", "internet")

# --- frozen dev/test split (ticket 260) ---------------------------------------
# Every row carries a `split` field: "dev" or "test". The assignment is a
# deterministic function of the row id alone (the rule is verified per row by
# _check_row):
#
#     split = "dev"  iff  int(sha256(f"implicit-split:{SPLIT_SALT}:<id>").hexdigest(), 16) % 2 == 0
#     split = "test" otherwise
#
# SPLIT_SALT = "7" was chosen so the assignment is stratified by kind over the
# full 100-row set (v1 + v2): every kind's dev share lands in
# SPLIT_PER_KIND_RANGE and the overall split is 50 dev / 50 test. The `test`
# rows are the FROZEN judge set: prompt and gate tuning may look ONLY at `dev`
# rows. The rule and the frozen test-row ids are recorded in
# docs/SUFFICIENCY-EXPERIMENT-RESULTS-2026-11-27.md ("Dev/test split").
SPLIT_SALT = "7"
SPLIT_VALUES = ("dev", "test")
SPLIT_PER_KIND_RANGE = (0.30, 0.70)  # allowed dev share within one kind
SPLIT_OVERALL_RANGE = (0.40, 0.60)  # allowed dev share over all rows
# the balance is only meaningful at a scale where 50/50 is attainable
SPLIT_MIN_ROWS_FOR_BALANCE = 20


def split_for(rid: str) -> str:
    """The ticket-260 split rule: deterministic hash parity of the row id."""
    digest = hashlib.sha256(f"implicit-split:{SPLIT_SALT}:{rid}".encode()).hexdigest()
    return "dev" if int(digest, 16) % 2 == 0 else "test"


def split_balance_errors(rows: list[dict[str, Any]]) -> list[str]:
    """Stratification check: per-kind and overall dev shares within the allowed ranges."""
    errors: list[str] = []
    if len(rows) < SPLIT_MIN_ROWS_FOR_BALANCE:
        return errors
    kind_total: Counter[str] = Counter()
    kind_dev: Counter[str] = Counter()
    for row in rows:
        kind = row.get("kind")
        if kind not in KINDS:
            continue
        kind_total[kind] += 1
        if row.get("split") == "dev":
            kind_dev[kind] += 1
    lo_k, hi_k = SPLIT_PER_KIND_RANGE
    for kind, n in sorted(kind_total.items()):
        share = kind_dev[kind] / n
        if not lo_k <= share <= hi_k:
            errors.append(
                f"split balance: kind {kind!r} dev share {share:.2f} outside [{lo_k}, {hi_k}]"
            )
    n_dev = sum(1 for row in rows if row.get("split") == "dev")
    share_all = n_dev / len(rows)
    lo_a, hi_a = SPLIT_OVERALL_RANGE
    if not lo_a <= share_all <= hi_a:
        errors.append(f"split balance: overall dev share {share_all:.2f} outside [{lo_a}, {hi_a}]")
    return errors


TEXT_EXTS = {".md", ".txt", ".csv", ".json"}
PARSED_EXTS = {".docx", ".pptx", ".xlsx", ".pdf", ".html", ".htm"}


def extract_text(path: Path) -> str | None:
    """Return the text of a corpus file, or None if no parser can handle it.

    Plain-text formats are read directly; office/PDF/HTML formats go through
    markitdown when it is importable.
    """
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    if ext in PARSED_EXTS:
        try:
            from markitdown import MarkItDown
        except ImportError:
            return None
        try:
            return MarkItDown().convert(str(path)).text_content or ""
        except Exception:
            return None
    return None


def _check_row(row: dict[str, Any], root: Path, texts: dict[str, str | None]) -> list[str]:
    errors: list[str] = []
    rid = row.get("id", "<no id>")

    for key in REQUIRED_KEYS:
        if key not in row:
            errors.append(f"{rid}: missing key {key!r}")
    if errors:
        return errors

    if not ID_RE.match(row["id"]):
        errors.append(f"{rid}: id does not match <tag>-NNN")
    if row["kind"] not in KINDS:
        errors.append(f"{rid}: unknown kind {row['kind']!r}")

    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{rid}: empty question")
    else:
        n_words = len(question.split())
        if not MIN_QUESTION_WORDS <= n_words <= MAX_QUESTION_WORDS:
            errors.append(
                f"{rid}: question has {n_words} words "
                f"(want {MIN_QUESTION_WORDS}-{MAX_QUESTION_WORDS})"
            )
        if not question.rstrip().endswith("?"):
            errors.append(f"{rid}: question does not end with a question mark")
        low = question.lower()
        for word in FORBIDDEN_QUESTION_WORDS:
            if re.search(rf"\b{word}\b", low):
                errors.append(f"{rid}: question names a source via {word!r}")

    for key in ("doc_fact", "why", "ideal_answer"):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{rid}: {key!r} must be a non-empty string")

    doc_file = row["doc_file"]
    if not isinstance(doc_file, str) or not doc_file:
        errors.append(f"{rid}: doc_file must be a non-empty string")
    else:
        if doc_file.startswith("/") or ".." in Path(doc_file).parts:
            errors.append(f"{rid}: doc_file must be a relative path under the corpus root")
        else:
            target = root / doc_file
            if not target.is_file():
                errors.append(f"{rid}: doc_file {doc_file!r} not found under corpus root")
            elif doc_file not in texts:
                texts[doc_file] = extract_text(target)

    if row["kind"].startswith("needs_web"):
        if row["web_needed"] is not True:
            errors.append(f"{rid}: {row['kind']} rows must have web_needed=true")
        wf = row["web_fact"]
        if not isinstance(wf, str) or not wf.strip():
            errors.append(f"{rid}: needs_web rows must have a non-empty web_fact")
    else:
        if row["web_needed"] is not False:
            errors.append(f"{rid}: docs_only_control rows must have web_needed=false")
        if row["web_fact"] is not None:
            errors.append(f"{rid}: docs_only_control rows must have web_fact=null")

    if row["kind"] == "needs_web_stale":
        sm = row["stale_marker"]
        if not isinstance(sm, str) or not sm.strip():
            errors.append(f"{rid}: needs_web_stale rows must set stale_marker")
    elif not isinstance(row["stale_marker"], (str, type(None))):
        errors.append(f"{rid}: stale_marker must be a string or null")

    _check_split(rid, row["split"], errors)

    evidence = row["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        errors.append(f"{rid}: evidence must be a non-empty string")
    else:
        if len(evidence) > MAX_EVIDENCE_CHARS:
            errors.append(f"{rid}: evidence is {len(evidence)} chars (max {MAX_EVIDENCE_CHARS})")
        if isinstance(doc_file, str) and doc_file in texts:
            text = texts[doc_file]
            if text is None:
                errors.append(f"{rid}: no parser available for {doc_file!r}; evidence not checked")
            elif evidence not in text:
                errors.append(f"{rid}: evidence snippet not found in {doc_file}")

    return errors


def _check_split(rid: str, split: Any, errors: list[str]) -> None:
    """The frozen split (ticket 260): the value and its hash-rule assignment."""
    if split not in SPLIT_VALUES:
        errors.append(f"{rid}: split must be one of {SPLIT_VALUES}, got {split!r}")
    elif split != split_for(rid):
        errors.append(
            f"{rid}: split {split!r} does not match the ticket-260 hash rule "
            f"(expected {split_for(rid)!r})"
        )


def validate_rows(
    rows: list[dict[str, Any]],
    root: Path,
    *,
    check_evidence: bool = True,
    expected_counts: dict[str, int] | None = None,
    total_rows: int | None = None,
    min_files: int | None = None,
) -> list[str]:
    """Validate a question set. Returns a list of error strings (empty = pass)."""
    errors: list[str] = []
    counts = expected_counts if expected_counts is not None else EXPECTED_KIND_COUNTS
    want_total = total_rows if total_rows is not None else sum(counts.values())
    want_files = min_files if min_files is not None else MIN_DISTINCT_FILES

    if len(rows) != want_total:
        errors.append(f"expected {want_total} rows, found {len(rows)}")

    kind_counts = Counter(row.get("kind") for row in rows)
    for kind, want in counts.items():
        got = kind_counts.get(kind, 0)
        if got != want:
            errors.append(f"kind {kind!r}: expected {want} rows, found {got}")

    ids = [row.get("id") for row in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    for i in sorted(d for d in dupes if isinstance(d, str)):
        errors.append(f"duplicate id {i!r}")

    texts: dict[str, str | None] = {}
    for row in rows:
        if check_evidence:
            row_errors = _check_row(row, root, texts)
        else:
            row_errors = _check_row_no_evidence(row, root)
        errors.extend(row_errors)

    files = {row.get("doc_file") for row in rows if isinstance(row.get("doc_file"), str)}
    if len(files) < want_files:
        errors.append(f"only {len(files)} distinct doc_file values (want at least {want_files})")

    errors.extend(split_balance_errors(rows))
    return errors


def _check_row_no_evidence(row: dict[str, Any], root: Path) -> list[str]:
    """Row checks with the evidence-in-file step disabled."""
    errors: list[str] = []
    rid = row.get("id", "<no id>")

    for key in REQUIRED_KEYS:
        if key not in row:
            errors.append(f"{rid}: missing key {key!r}")
    if errors:
        return errors

    if not ID_RE.match(row["id"]):
        errors.append(f"{rid}: id does not match <tag>-NNN")
    if row["kind"] not in KINDS:
        errors.append(f"{rid}: unknown kind {row['kind']!r}")

    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{rid}: empty question")
    else:
        n_words = len(question.split())
        if not MIN_QUESTION_WORDS <= n_words <= MAX_QUESTION_WORDS:
            errors.append(
                f"{rid}: question has {n_words} words "
                f"(want {MIN_QUESTION_WORDS}-{MAX_QUESTION_WORDS})"
            )
        if not question.rstrip().endswith("?"):
            errors.append(f"{rid}: question does not end with a question mark")
        low = question.lower()
        for word in FORBIDDEN_QUESTION_WORDS:
            if re.search(rf"\b{word}\b", low):
                errors.append(f"{rid}: question names a source via {word!r}")

    for key in ("doc_fact", "why", "ideal_answer"):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{rid}: {key!r} must be a non-empty string")

    doc_file = row["doc_file"]
    if not isinstance(doc_file, str) or not doc_file:
        errors.append(f"{rid}: doc_file must be a non-empty string")
    elif doc_file.startswith("/") or ".." in Path(doc_file).parts:
        errors.append(f"{rid}: doc_file must be a relative path under the corpus root")
    elif not (root / doc_file).is_file():
        errors.append(f"{rid}: doc_file {doc_file!r} not found under corpus root")

    if row["kind"].startswith("needs_web"):
        if row["web_needed"] is not True:
            errors.append(f"{rid}: {row['kind']} rows must have web_needed=true")
        wf = row["web_fact"]
        if not isinstance(wf, str) or not wf.strip():
            errors.append(f"{rid}: needs_web rows must have a non-empty web_fact")
    else:
        if row["web_needed"] is not False:
            errors.append(f"{rid}: docs_only_control rows must have web_needed=false")
        if row["web_fact"] is not None:
            errors.append(f"{rid}: docs_only_control rows must have web_fact=null")

    if row["kind"] == "needs_web_stale":
        sm = row["stale_marker"]
        if not isinstance(sm, str) or not sm.strip():
            errors.append(f"{rid}: needs_web_stale rows must set stale_marker")
    elif not isinstance(row["stale_marker"], (str, type(None))):
        errors.append(f"{rid}: stale_marker must be a string or null")

    _check_split(rid, row["split"], errors)

    evidence = row["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        errors.append(f"{rid}: evidence must be a non-empty string")
    elif len(evidence) > MAX_EVIDENCE_CHARS:
        errors.append(f"{rid}: evidence is {len(evidence)} chars (max {MAX_EVIDENCE_CHARS})")

    return errors


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                rows.append({"id": f"line-{n}", "kind": None, "_json_error": str(e)})
                continue
            if not isinstance(row, dict):
                rows.append({"id": f"line-{n}", "kind": None, "_json_error": "not an object"})
            else:
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", required=True, type=Path, help="path to the question-set JSONL file"
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="corpus root; doc_file values resolve under it",
    )
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="skip the evidence-in-file check (when parsers are unavailable)",
    )
    parser.add_argument(
        "--expected-counts",
        default=None,
        help="override kind counts, e.g. 'needs_web_stale=2,docs_only_control=4' "
        "(overrides the built-in 20/15/10/15 balance and the 60-row total)",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=None,
        help="override the minimum distinct doc_file count",
    )
    parser.add_argument(
        "--also",
        type=Path,
        default=None,
        help="a second question-set file (usually v1 when checking v2, or vice "
        "versa): it is validated too, and the frozen dev/test split balance is "
        "checked over the union of the two sets",
    )
    parser.add_argument(
        "--expected-counts-2",
        default=None,
        help="kind counts for the --also file, same syntax as --expected-counts",
    )
    args = parser.parse_args(argv)

    if not args.set.is_file():
        print(f"FAIL: set file not found: {args.set}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"FAIL: corpus root not found: {args.root}", file=sys.stderr)
        return 1
    if args.also is not None and not args.also.is_file():
        print(f"FAIL: --also file not found: {args.also}", file=sys.stderr)
        return 1

    def _parse_counts(value: str | None, flag: str) -> dict[str, int] | None:
        if not value:
            return None
        counts: dict[str, int] = {}
        for part in value.split(","):
            kind, _, number = part.partition("=")
            if not kind.strip() or not number.strip().isdigit():
                print(f"FAIL: bad {flag} entry {part!r}", file=sys.stderr)
                raise SystemExit(1)
            counts[kind.strip()] = int(number)
        return counts

    try:
        counts = _parse_counts(args.expected_counts, "--expected-counts")
        counts2 = _parse_counts(args.expected_counts_2, "--expected-counts-2")
    except SystemExit:
        return 1

    all_errors: list[str] = []
    union_rows: list[dict[str, Any]] = []

    rows = load_rows(args.set)
    union_rows.extend(rows)
    errors = validate_rows(
        rows,
        args.root,
        check_evidence=not args.skip_evidence,
        expected_counts=counts,
        total_rows=sum(counts.values()) if counts else None,
        min_files=args.min_files,
    )
    kind_counts = Counter(r.get("kind") for r in rows if isinstance(r.get("kind"), str))
    files = {r.get("doc_file") for r in rows if isinstance(r.get("doc_file"), str)}
    print(
        f"{args.set.name}: rows={len(rows)} kinds={dict(kind_counts)} distinct_files={len(files)}"
    )
    all_errors.extend(f"({args.set.name}) {e}" for e in errors)

    if args.also is not None:
        rows2 = load_rows(args.also)
        union_rows.extend(rows2)
        errors2 = validate_rows(
            rows2,
            args.root,
            check_evidence=not args.skip_evidence,
            expected_counts=counts2,
            total_rows=sum(counts2.values()) if counts2 else None,
            min_files=None,
        )
        kind_counts2 = Counter(r.get("kind") for r in rows2 if isinstance(r.get("kind"), str))
        print(f"{args.also.name}: rows={len(rows2)} kinds={dict(kind_counts2)}")
        all_errors.extend(f"({args.also.name}) {e}" for e in errors2)
        # the frozen split is a property of the full (union) set
        balance = split_balance_errors(union_rows)
        if balance:
            dev = sum(1 for r in union_rows if r.get("split") == "dev")
            print(f"union: rows={len(union_rows)} dev={dev} test={len(union_rows) - dev}")
            all_errors.extend(f"(union) {e}" for e in balance)

    if all_errors:
        print(f"FAIL: {len(all_errors)} problem(s):", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
