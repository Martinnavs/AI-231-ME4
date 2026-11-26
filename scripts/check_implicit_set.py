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
)
ID_RE = re.compile(r"^[a-z][a-z0-9]*-\d{3}$")
# A question must read as an ordinary spoken question: it may not name the
# sources (corpus documents or the web) in any form.
FORBIDDEN_QUESTION_WORDS = ("document", "file", "note", "web", "internet")

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
    args = parser.parse_args(argv)

    if not args.set.is_file():
        print(f"FAIL: set file not found: {args.set}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"FAIL: corpus root not found: {args.root}", file=sys.stderr)
        return 1

    counts = None
    if args.expected_counts:
        counts = {}
        for part in args.expected_counts.split(","):
            kind, _, value = part.partition("=")
            if not kind.strip() or not value.strip().isdigit():
                print(f"FAIL: bad --expected-counts entry {part!r}", file=sys.stderr)
                return 1
            counts[kind.strip()] = int(value)

    rows = load_rows(args.set)
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
    print(f"rows={len(rows)} kinds={dict(kind_counts)} distinct_files={len(files)}")
    if errors:
        print(f"FAIL: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
