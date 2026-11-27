"""Unsupported-type files are spoken once, as a count, never by file name (ticket 252)."""

from __future__ import annotations

import json

from voice_pipeline import doc_route


def _status(failures: list[dict[str, object]]) -> str:
    return json.dumps({"folder": "x", "files": [], "failures": failures})


def _row(name: str, status: str, reason: str = "") -> dict[str, object]:
    return {"name": name, "status": status, "reason": reason, "mtime": 1, "size": 1}


def test_many_unsupported_files_are_one_sentence_without_names() -> None:
    rows = [
        _row(f"data/table_{i}.json", "unsupported", "unsupported file extension: .json")
        for i in range(9)
    ]
    prefix, _ = doc_route.new_failures_prefix(_status(rows), {})
    assert prefix == "9 files are in a format I can't read."
    assert ".json" not in prefix and "table_" not in prefix


def test_one_unsupported_file_and_a_real_failure_keep_their_own_wording() -> None:
    rows = [
        _row("a/note.odt", "unsupported"),
        _row("b/broken.pdf", "unreadable", "pdf parse error"),
    ]
    prefix, _ = doc_route.new_failures_prefix(_status(rows), {})
    assert "1 file is in a format I can't read." in prefix
    assert "I couldn't read b/broken.pdf: pdf parse error." in prefix


def test_the_notice_is_spoken_once_until_the_set_changes() -> None:
    seen: dict[str, tuple[object, ...]] = {}
    rows = [_row("a.odt", "unsupported"), _row("b.eml", "unsupported")]
    first, _ = doc_route.new_failures_prefix(_status(rows), seen)
    again, _ = doc_route.new_failures_prefix(_status(rows), seen)
    assert first == "2 files are in a format I can't read." and again == ""
    changed, _ = doc_route.new_failures_prefix(_status([*rows, _row("c.xml", "unsupported")]), seen)
    assert changed == "3 files are in a format I can't read."
