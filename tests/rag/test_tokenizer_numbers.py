"""Tests for sp1-T3: number tokenization, FTS5 pretok header dedup, and schema rebuild."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from doc_chat.corpus import tokenize
from voice_agent.rag import fts5_store as fts5_store_mod
from voice_agent.rag.docindex import Bm25Retriever, Chunk
from voice_agent.rag.fts5 import Fts5Retriever, _pretok_row
from voice_agent.rag.fts5_store import DEFAULT_TOKENIZER, SCHEMA_VERSION, Fts5Store


def _make_chunk(idx: int, source: str, header: str, text: str) -> Chunk:
    return Chunk(
        id=f"{source}#{idx}",
        file_sha256=f"fsha-{source}",
        chunk_sha256=f"csha-{idx}",
        position=idx,
        page=None,
        section=header,
        file_date="2026-01-15",
        source=source,
        header=header,
        text=text,
    )


def test_thousands_commas_and_decimals_tokenization() -> None:
    assert tokenize("1,599") == ["1599"]
    assert tokenize("1,234,567.89") == ["1234567.89"]
    assert tokenize(",234,567.89") == ["234567.89"]
    assert tokenize("3,150") == tokenize("3150") == ["3150"]
    assert tokenize("1,5") == ["1", "5"]
    assert tokenize("1, 599") == ["1", "599"]
    assert tokenize("2024,2025") == ["2024", "2025"]
    assert tokenize("1,5990") == ["1", "5990"]
    assert tokenize("12.5") == ["12.5"]
    assert tokenize("v1.2.3") == ["v1.2.3"]


def test_dots_non_digits_and_currency_percent_separators() -> None:
    assert tokenize("e.g.") == ["e", "g"]
    assert tokenize("file.name") == ["file", "name"]
    assert tokenize("file.123") == ["file", "123"]
    assert tokenize("1.name") == ["1", "name"]
    assert tokenize("$1,599.50") == ["1599.50"]
    assert tokenize("12.5%") == ["12.5"]
    assert tokenize("1,234.56%") == ["1234.56"]


def test_stemming_stopwords_and_apostrophes_unchanged() -> None:
    assert tokenize("WHAT IS A HORCRUX?") == ["horcrux"]
    assert tokenize("The Horcruxes") == ["horcrux"]
    assert tokenize("Dumbledore's") == ["dumbledore"]
    assert tokenize("don't stop") == ["don't", "stop"]
    assert tokenize("Boxes's churches wishes passes cats\u2019 running walked the and about") == [
        "box",
        "church",
        "wish",
        "pass",
        "cat",
        "runn",
        "walk",
    ]


def test_tokenize_linear_time_on_1mb_commas_and_digits() -> None:
    repeating = "1,234,567.89 " * 80_000
    assert len(repeating) > 1_000_000
    t0 = time.monotonic()
    tokens = tokenize(repeating)
    elapsed = time.monotonic() - t0
    assert len(tokens) == 80_000
    assert tokens[0] == "1234567.89"
    assert elapsed < 5.0

    adversarial = ("999," * 260_000) + "1234"
    assert len(adversarial) > 1_000_000
    t1 = time.monotonic()
    adv_tokens = tokenize(adversarial)
    adv_elapsed = time.monotonic() - t1
    assert adv_tokens[-1] == "1234"
    assert adv_elapsed < 5.0


def test_pretok_row_excludes_header_tokens_from_text_column() -> None:
    chunk = _make_chunk(0, "guide.md", "Guide > Firmware Update", "Reboot the router now.")
    chunk_id, _key, text_col, header_col = _pretok_row(chunk)
    assert chunk_id == "guide.md#0"
    assert header_col == "guide firmware update"
    assert text_col == "reboot router now"
    for tok in header_col.split():
        assert tok not in text_col.split()


def test_bm25_and_fts5_pick_up_number_tokenization(tmp_path: Path) -> None:
    chunks = [
        _make_chunk(0, "budget.md", "budget.md > CapEx", "Annual budget is $3,150 for servers."),
        _make_chunk(1, "spec.md", "spec.md > Voltage", "Operating threshold is 12.5 V nominal."),
        _make_chunk(2, "noise.md", "noise.md > Misc", "Item 1 and item 599 and value 5."),
    ]
    bm25 = Bm25Retriever()
    bm25.build(chunks)
    assert [idx for idx, _ in bm25.rank("3150", 5)] == [0]
    assert [idx for idx, _ in bm25.rank("3,150", 5)] == [0]
    assert [idx for idx, _ in bm25.rank("12.5", 5)] == [1]

    db_path = tmp_path / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_path, tmp_path, tokenizer="pretok") as fts:
        fts.build(chunks)
        assert [idx for idx, _ in fts.rank("3150", 5)] == [0]
        assert [idx for idx, _ in fts.rank("3,150", 5)] == [0]
        assert [idx for idx, _ in fts.rank("12.5", 5)] == [1]


def test_schema_version_bumped_and_stale_v1_db_rebuilt_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert SCHEMA_VERSION == "2"

    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"
    chunks = [
        _make_chunk(0, "prices.md", "prices.md > Hardware", "Unit cost is $1,599."),
        _make_chunk(1, "other.md", "other.md > Notes", "Part code 599 is deprecated."),
    ]

    with Fts5Retriever(db_path, root, tokenizer="pretok") as seed:
        seed.build(chunks)

    # Simulate a pre-T3 schema_version='1' database containing old split tokens ("1 599").
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        conn.execute("UPDATE fts SET all_tokens = 'unit cost 1 599' WHERE rowid = 1")
        conn.commit()
    finally:
        conn.close()

    rebuild_calls = 0
    orig_create = fts5_store_mod.Fts5Store._create_and_init_disk_db

    def _counting_create(
        self: Fts5Store,
        dfd: int,
        db_path_arg: Path,
        db_name: str,
        *,
        validate: bool,
    ) -> sqlite3.Connection:
        nonlocal rebuild_calls
        rebuild_calls += 1
        return orig_create(self, dfd, db_path_arg, db_name, validate=validate)

    monkeypatch.setattr(fts5_store_mod.Fts5Store, "_create_and_init_disk_db", _counting_create)

    with Fts5Retriever(db_path, root, tokenizer="pretok") as rebuilt:
        assert rebuild_calls == 1
        rebuilt.build(chunks)
        assert [idx for idx, _ in rebuilt.rank("1,599", 5)] == [0]
        assert [idx for idx, _ in rebuilt.rank("1599", 5)] == [0]
        assert [idx for idx, _ in rebuilt.rank("599", 5)] == [1]

        assert rebuilt._store is not None
        assert rebuilt._store._conn is not None
        meta = dict(rebuilt._store._conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["schema_version"] == "2"

        # Verify stored FTS text column has body tokens only (not header tokens).
        stored_row = rebuilt._store._conn.execute(
            "SELECT all_tokens, header_tokens FROM fts WHERE rowid = 1"
        ).fetchone()
        assert stored_row == ("unit cost 1599", "price hardware")

    # Reopening the rebuilt DB does not trigger another rebuild.
    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as reopened:
        assert rebuild_calls == 1
        assert [cid for cid, _ in reopened.query('"1599"')] == ["prices.md#0"]


def test_thousands_group_cap_bounds_token_lengths_and_stays_linear() -> None:
    assert tokenize("1,599") == ["1599"]
    assert tokenize("1,234,567,890") == ["1234567890"]

    seven_groups = "101,102,103,104,105,106,107"
    seven_tokens = tokenize(seven_groups)
    assert len(seven_tokens) > 1
    assert seven_tokens == ["101102103104105106", "107"]
    assert all(len(tok) <= 18 for tok in seven_tokens)

    one_mb_commas = ("123," * 260_000) + "456"
    assert len(one_mb_commas) > 1_000_000
    t0 = time.monotonic()
    tokens = tokenize(one_mb_commas)
    elapsed = time.monotonic() - t0
    assert tokens
    assert max(len(tok) for tok in tokens) <= 18
    assert elapsed < 5.0
