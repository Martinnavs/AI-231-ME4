"""Tests for scripts/check_implicit_set.py with a tiny synthetic corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts))

import check_implicit_set as cis  # noqa: E402


def _corpus(tmp_path: Path) -> Path:
    """Tiny synthetic corpus: three plain-text files in two subfolders."""
    root = tmp_path / "corpus"
    (root / "agency").mkdir(parents=True)
    (root / "school").mkdir()
    (root / "agency" / "rates.txt").write_text(
        "The standard lodging rate is $110 per night for FY2026.\n"
        "The state's capital is Santa Fe.\n",
        encoding="utf-8",
    )
    (root / "agency" / "pay.txt").write_text(
        "The GS-12 step 5 salary is $103,049 effective January 2026.\n",
        encoding="utf-8",
    )
    (root / "school" / "guide.txt").write_text(
        "The minimum font size should be 12 points.\n"
        "Finals run December 7-11, Monday through Friday.\n",
        encoding="utf-8",
    )
    return root


def _row(**over) -> dict:
    base = {
        "id": "imp-001",
        "kind": "docs_only_control",
        "question": "What is the standard lodging rate per night?",
        "doc_fact": "Standard lodging rate is $110 per night.",
        "doc_file": "agency/rates.txt",
        "web_needed": False,
        "web_fact": None,
        "why": "Settled rate in the document.",
        "ideal_answer": "The standard rate is $110 a night.",
        "stale_marker": None,
        "evidence": "The standard lodging rate is $110 per night",
    }
    base.update(over)
    return base


def _small_set(root: Path) -> list[dict]:
    """Six rows matching small expected counts (1/1/1/3) across 3 files."""
    return [
        _row(id="imp-001"),
        _row(
            id="imp-002",
            kind="needs_web_stale",
            question="What is the lodging rate next fiscal year?",
            doc_fact="FY2026 standard lodging rate is $110 per night.",
            web_needed=True,
            web_fact="Next fiscal year rate - verify live",
            why="The document only covers FY2026.",
            ideal_answer="FY2026 shows $110. For the next fiscal year verify live.",
            stale_marker="FY2026",
            evidence="The standard lodging rate is $110 per night for FY2026.",
        ),
        _row(
            id="imp-003",
            kind="needs_web_missing_half",
            question="What is the capital of the state involved?",
            doc_fact="The document concerns the state (capital not named).",
            web_needed=True,
            web_fact="Capital of the state (Santa Fe)",
            why="The state is identifiable but its capital is outside the corpus.",
            ideal_answer="That is the state budget; its capital is Santa Fe (web).",
            evidence="The state's capital is Santa Fe.",
        ),
        _row(
            id="imp-004",
            kind="needs_web_conflict",
            question="When do finals run this semester?",
            doc_fact="Proposed calendar lists finals December 7-11.",
            doc_file="school/guide.txt",
            web_needed=True,
            web_fact="Confirmed exam dates - verify live",
            why="The dates are from a proposed calendar and may change.",
            ideal_answer="Finals are proposed for December 7-11; verify the finalized dates live.",
            stale_marker=None,
            evidence="Finals run December 7-11",
        ),
        _row(
            id="imp-005",
            kind="docs_only_control",
            question="What is the minimum font size for a paper?",
            doc_fact="Minimum font size is 12 points.",
            doc_file="school/guide.txt",
            why="Settled formatting rule in the document.",
            ideal_answer="The minimum font size is 12 points.",
            evidence="The minimum font size should be 12 points.",
        ),
        _row(
            id="imp-006",
            kind="docs_only_control",
            question="How much does a GS-12 step 5 make now?",
            doc_fact="GS-12 step 5 is $103,049 under the 2026 table in force.",
            doc_file="agency/pay.txt",
            why="The 2026 table is the schedule currently in force.",
            ideal_answer="The current table puts GS-12 step 5 at $103,049.",
            stale_marker="January 2026",
            evidence="The GS-12 step 5 salary is $103,049",
        ),
    ]


SMALL_COUNTS = {
    "needs_web_stale": 1,
    "needs_web_missing_half": 1,
    "needs_web_conflict": 1,
    "docs_only_control": 3,
}


def _validate(rows, root):
    return cis.validate_rows(rows, root, expected_counts=SMALL_COUNTS, total_rows=6, min_files=3)


def test_clean_set_passes(tmp_path: Path):
    root = _corpus(tmp_path)
    assert _validate(_small_set(root), root) == []


def test_evidence_not_found(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["evidence"] = "a snippet that is not in any corpus file"
    errors = _validate(rows, root)
    assert any("imp-001" in e and "evidence snippet not found" in e for e in errors)


def test_skip_evidence_allows_missing_snippet(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["evidence"] = "a snippet that is not in any corpus file"
    errors = cis.validate_rows(
        rows, root, check_evidence=False, expected_counts=SMALL_COUNTS, total_rows=6, min_files=3
    )
    assert errors == []


def test_duplicate_id(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[1]["id"] = "imp-001"
    errors = _validate(rows, root)
    assert any("duplicate id" in e for e in errors)


def test_forbidden_word_in_question(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["question"] = "What is the lodging rate in the document per night?"
    errors = _validate(rows, root)
    assert any("document" in e for e in errors)


def test_question_too_short(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["question"] = "What is the rate?"
    errors = _validate(rows, root)
    assert any("words" in e and "imp-001" in e for e in errors)


def test_missing_doc_file(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["doc_file"] = "agency/does_not_exist.txt"
    errors = _validate(rows, root)
    assert any("not found under corpus root" in e for e in errors)


def test_wrong_kind_count(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["kind"] = "needs_web_conflict"
    rows[0]["web_needed"] = True
    rows[0]["web_fact"] = "verify live"
    errors = _validate(rows, root)
    assert any("needs_web_conflict" in e and "expected 1" in e for e in errors)
    assert any("docs_only_control" in e and "expected 3" in e for e in errors)


def test_needs_web_row_without_web_fact(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[1]["web_fact"] = None
    errors = _validate(rows, root)
    assert any("imp-002" in e and "web_fact" in e for e in errors)


def test_stale_row_without_stale_marker(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[1]["stale_marker"] = None
    errors = _validate(rows, root)
    assert any("imp-002" in e and "stale_marker" in e for e in errors)


def test_control_row_with_web_fact_fails(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["web_fact"] = "should be null"
    errors = _validate(rows, root)
    assert any("imp-001" in e and "web_fact=null" in e for e in errors)


def test_not_a_question(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    rows[0]["question"] = "The standard lodging rate per night is"
    errors = _validate(rows, root)
    assert any("question mark" in e for e in errors)


def test_too_few_distinct_files(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    for r in rows:
        r["doc_file"] = "agency/rates.txt"
        r["evidence"] = "The standard lodging rate is $110 per night"
    errors = _validate(rows, root)
    assert any("distinct doc_file" in e for e in errors)


def _dump(rows: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def test_load_rows_roundtrip(tmp_path: Path):
    root = _corpus(tmp_path)
    rows = _small_set(root)
    path = tmp_path / "set.jsonl"
    _dump(rows, path)
    assert cis.load_rows(path) == rows


def test_main_cli_pass_and_fail(tmp_path: Path, capsys):
    root = _corpus(tmp_path)
    good = tmp_path / "good.jsonl"
    _dump(_small_set(root), good)
    counts = (
        "--expected-counts",
        "needs_web_stale=1,needs_web_missing_half=1,needs_web_conflict=1,docs_only_control=3",
        "--min-files",
        "3",
    )
    # Evidence check on: the plain-text corpus must verify cleanly.
    assert cis.main(["--set", str(good), "--root", str(root), *counts]) == 0
    assert "PASS" in capsys.readouterr().out

    # Evidence check off: a bogus snippet is no longer caught.
    rows = _small_set(root)
    rows[0]["evidence"] = "a snippet that is not in any corpus file"
    loose = tmp_path / "loose.jsonl"
    _dump(rows, loose)
    assert cis.main(["--set", str(loose), "--root", str(root), *counts, "--skip-evidence"]) == 0
    assert cis.main(["--set", str(loose), "--root", str(root), *counts]) == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json at all\n", encoding="utf-8")
    assert cis.main(["--set", str(bad), "--root", str(root), "--skip-evidence"]) == 1
    assert "FAIL" in capsys.readouterr().err

    assert cis.main(["--set", str(tmp_path / "missing.jsonl"), "--root", str(root)]) == 1
    assert cis.main(["--set", str(good), "--root", str(tmp_path / "nope")]) == 1
    assert cis.main(["--set", str(good), "--root", str(root), "--expected-counts", "oops"]) == 1
