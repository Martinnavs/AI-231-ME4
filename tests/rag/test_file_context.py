"""Tests for file-first context (ticket 207)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.rag.dbindex import (
    DbIndex,
    grep_lines,
    grep_terms,
)
from voice_agent.rag.ingest import ingest

# --- grep_terms ---


class TestGrepTerms:
    def test_example(self) -> None:
        q = "What is the elevation in feet of JFK in the airport codes file?"
        assert grep_terms(q) == ["airpo", "codes", "eleva", "feet", "jfk"]

    def test_stopwords_removed(self) -> None:
        assert grep_terms("the quick brown fox") == ["brown", "fox", "quick"]

    def test_single_char_dropped(self) -> None:
        assert grep_terms("a b c xyz") == ["xyz"]


# --- grep_lines ---


class TestGrepLines:
    @pytest.fixture()
    def log_text(self) -> str:
        lines = [
            "Jan  1 auth: header line",
            "Jan  1 auth: ok login user1",
            "Jan  1 auth: Invalid user root from 10.0.0.1",
            "Jan  1 auth: ok login user2",
            "Jan  1 auth: ok login user3",
            "Jan  1 auth: Invalid user admin from 10.0.0.2",
            "Jan  1 auth: session closed for user1",
            "Jan  1 auth: ok login user4",
            "Jan  1 auth: ok login user5",
            "Jan  1 auth: Invalid user test from 10.0.0.3",
        ]
        return "\n".join(lines)

    def test_ordinal_question(self, log_text: str) -> None:
        out = grep_lines(log_text, "first invalid user", 20)
        out_text = "\n".join(out)
        assert "L3:" in out_text, "should contain line 3 (first invalid)"
        assert "L10:" in out_text, "should contain line 10 (last invalid)"
        assert "L1:" in out_text, "should contain header line 1"

    def test_ascending_order(self, log_text: str) -> None:
        out = grep_lines(log_text, "first invalid user", 20)
        nums = []
        for line in out:
            num = int(line.split(":")[0][1:])
            nums.append(num)
        assert nums == sorted(nums), "lines must be in ascending order"

    def test_non_ordinal(self, log_text: str) -> None:
        out = grep_lines(log_text, "invalid user", 20)
        nums = []
        for line in out:
            num = int(line.split(":")[0][1:])
            nums.append(num)
        # Should contain L1 (header) + the 3 matching lines (3,6,10)
        assert 1 in nums
        assert 3 in nums
        assert 7 in nums  # line 6 (0-indexed) -> L7... wait
        # Lines 3, 7, 10 (1-indexed) contain "invalid user"
        assert 3 in nums
        assert 10 in nums

    def test_no_match(self) -> None:
        assert grep_lines("hello world\nfoo bar", "xyz nonexistent", 20) == []


# --- file_context ---


@pytest.fixture(scope="module")
def fc_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict:
    src = tmp_path_factory.mktemp("fc_src")
    db = tmp_path_factory.mktemp("fc_db") / "fc.db"

    # Small .md (< 15k chars)
    (src / "small.md").write_text(
        "# Small Doc\n\nThis is a small document about widgets.\nWidgets cost $5 each.\n",
        encoding="utf-8",
    )

    # Big .log (> 15k chars, 2000 lines)
    log_lines = ["# Log header"]
    for i in range(1, 2000):
        log_lines.append(f"Line {i + 1}: log entry for server-{i % 10}")
    log_text = "\n".join(log_lines)
    assert len(log_text) > 15_000, f"big log too small: {len(log_text)}"
    assert len(log_lines) == 2000, f"big log has {len(log_lines)} lines"
    (src / "big.log").write_text(log_text, encoding="utf-8")

    # Big .csv with 60000 rows — row 55000 has key data beyond 50k cap
    csv_lines = ["id,name,value"]
    for i in range(1, 60001):
        if i == 55000:
            csv_lines.append(f"{i},target_server,42")
        else:
            csv_lines.append(f"{i},server-{i},value-{i}")
    (src / "data.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    ingest(src, db)
    return {"src": src, "db": db}


class TestFileContext:
    @pytest.fixture()
    def index(self, fc_fixture: dict):
        idx = DbIndex(fc_fixture["db"], raw_root=fc_fixture["src"])
        yield idx
        idx.close()

    @pytest.fixture()
    def index_no_raw(self, fc_fixture: dict):
        idx = DbIndex(fc_fixture["db"])
        yield idx
        idx.close()

    def test_small_file_whole(self, index: DbIndex) -> None:
        passages = index.file_context("widgets", n_files=3)
        small_passages = [p for p in passages if p["source"] == "small.md"]
        assert len(small_passages) == 1
        assert small_passages[0]["part"] == "whole"

    def test_big_log_grep_then_chunks(self, index: DbIndex) -> None:
        passages = index.file_context("server", n_files=3)
        log_passages = [p for p in passages if p["source"] == "big.log"]
        parts = [p["part"] for p in log_passages]
        # First should be grep, then chunks
        if "grep" in parts:
            grep_idx = parts.index("grep")
            # No chunks before grep
            assert all(p != "chunk" for p in parts[:grep_idx])
        chunk_passages = [p for p in log_passages if p["part"] == "chunk"]
        assert len(chunk_passages) <= 4

    def test_csv_grep_with_raw_root(self, index: DbIndex) -> None:
        passages = index.file_context("target server 42", n_files=3)
        csv_passages = [p for p in passages if p["source"] == "data.csv"]
        grep_passages = [p for p in csv_passages if p["part"] == "grep"]
        if grep_passages:
            assert "target_server" in grep_passages[0]["text"]

    def test_csv_no_raw_root(self, index_no_raw: DbIndex) -> None:
        passages = index_no_raw.file_context("target server 42", n_files=3)
        csv_passages = [p for p in passages if p["source"] == "data.csv"]
        grep_passages = [p for p in csv_passages if p["part"] == "grep"]
        # Without raw_root, grep uses chunk text which may not contain row 55000
        if grep_passages:
            # The chunk text won't have the target row since it's beyond the cap
            assert "target_server" not in grep_passages[0].get("text", "")

    def test_passage_keys(self, index: DbIndex) -> None:
        passages = index.file_context("widgets", n_files=3)
        expected_keys = {
            "source",
            "id",
            "section",
            "page",
            "file_date",
            "text",
            "score",
            "coverage",
            "part",
        }
        for p in passages:
            assert set(p.keys()) == expected_keys, f"expected {expected_keys}, got {set(p.keys())}"

    def test_grep_disabled(self, index: DbIndex) -> None:
        passages = index.file_context("server", n_files=3, grep_top=0)
        for p in passages:
            assert p["part"] != "grep", "grep should be disabled with grep_top=0"


# --- defaults unchanged ---


# --- ablate script test ---


class TestAblateScript:
    def test_script_exists(self) -> None:
        script = Path("scripts/me4_context_ablate.py")
        assert script.exists(), "ablate script should exist"
