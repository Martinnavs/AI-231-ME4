"""Offline tests for voice_agent.rag.fts5 (Fts5Retriever)."""

from __future__ import annotations

import ast
import hashlib
import shutil
import sqlite3
from pathlib import Path
from typing import Literal

import pytest

from doc_chat.corpus import tokenize
from voice_agent.rag import fts5 as fts5_mod
from voice_agent.rag.docindex import HARD_CHUNKS_PER_FILE, Bm25Retriever, Chunk, DocIndex
from voice_agent.rag.fts5 import Fts5Retriever, _chunk_key
from voice_agent.rag.fts5_store import SyncStats
from voice_agent.rag.metrics import evaluate, format_metrics_table, load_golden

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"
FLOOR_TYPES = frozenset({"lexical", "numeric", "identifier"})


def _make_chunk(
    idx: int,
    source: str,
    header: str,
    text: str,
) -> Chunk:
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


def test_no_heavy_module_level_imports_and_key_matches_sha256() -> None:
    src_path = REPO_ROOT / "src" / "voice_agent" / "rag" / "fts5.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    forbidden = {"numpy", "rank_bm25"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = set(alias.name.split("."))
                assert not (parts & forbidden), f"forbidden module-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_parts = set((node.module or "").split("."))
            assert not (mod_parts & forbidden), f"forbidden module-level import: {node.module}"
            for alias in node.names:
                assert alias.name not in forbidden, f"forbidden module-level import: {alias.name}"

    expected = hashlib.sha256(b"manual.md > Intro\nhello world").hexdigest()
    assert _chunk_key("manual.md > Intro", "hello world") == expected


def test_invalid_init_params_and_empty_queries(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Fts5Retriever(None, tmp_path, tokenizer="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Fts5Retriever(None, tmp_path, header_weight=-1.0)
    with pytest.raises(ValueError):
        Fts5Retriever(None, tmp_path, header_weight=float("nan"))

    with Fts5Retriever(None, tmp_path) as retriever:
        assert retriever.status() == "fts5:memory"
        assert retriever.rank("alpha", 4) == []
        retriever.build([_make_chunk(0, "a.md", "a.md", "alpha beta")])
        assert retriever.rank("alpha", 0) == []
        assert retriever.rank("", 4) == []
        assert retriever.rank("the and or", 4) == []
        retriever.build([])
        assert retriever.rank("alpha", 4) == []


def test_pretok_apostrophe_tokenization_and_header_weighting(tmp_path: Path) -> None:
    chunks = [
        _make_chunk(0, "a.md", "a.md", "don't reboot without saving"),
        _make_chunk(1, "b.md", "b.md", "don Pedro wrote the guide"),
        _make_chunk(2, "c.md", "c.md > Firmware Update", "general instructions"),
        _make_chunk(3, "d.md", "d.md > Notes", "firmware update details"),
    ]
    with Fts5Retriever(None, tmp_path, tokenizer="pretok") as retriever:
        retriever.build(chunks)
        assert [idx for idx, _ in retriever.rank("don't", 10)] == [0]
        assert [idx for idx, _ in retriever.rank("don", 10)] == [1]
        # Header match gets boosted above body-only match.
        fw_hits = retriever.rank("firmware", 10)
        assert [idx for idx, _ in fw_hits] == [2, 3]


def test_docindex_search_keys_per_file_cap_no_overlap_and_coverage(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    # Create a file with >= 4 matching chunks and another file with 2 matching chunks,
    # plus a file sharing zero query tokens.
    para = "alpha calibration protocol step {} requires careful torque verification. "
    (docs / "big.md").write_text(
        "\n\n".join(f"## Sec {i}\n\n" + (para.format(i) * 18) for i in range(5)),
        encoding="utf-8",
    )
    (docs / "small.md").write_text(
        "## Overview\n\nalpha calibration summary.\n\n## Details\n\ntorque limits apply.",
        encoding="utf-8",
    )
    (docs / "unrelated.md").write_text(
        "## Kitchen\n\nbaking sourdough bread requires flour and water.",
        encoding="utf-8",
    )

    bm25_idx = DocIndex(docs)
    bm25_idx.refresh()

    db_path = tmp_path / "cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_path, tmp_path, tokenizer="pretok") as retriever:
        assert retriever.status() == "fts5:file"
        fts_idx = DocIndex(docs, retriever_factory=lambda: retriever)
        fts_idx.refresh()

        query = "alpha calibration torque"
        bm25_res = bm25_idx.search(query, k=4)
        fts_res = fts_idx.search(query, k=4)

        assert len(fts_res) == 4
        expected_keys = {
            "source",
            "id",
            "section",
            "page",
            "file_date",
            "text",
            "score",
            "coverage",
        }
        for hit in fts_res:
            assert set(hit.keys()) == expected_keys
            assert hit["source"] != "unrelated.md"

        # Per-file cap: soft cap 2, hard cap HARD_CHUNKS_PER_FILE; equal-scoring chunks of one
        # file may fill the top slots (score gate), the cap never exceeds the hard limit.
        source_counts: dict[str, int] = {}
        for hit in fts_res:
            source_counts[hit["source"]] = source_counts.get(hit["source"], 0) + 1
        assert source_counts.get("big.md", 0) <= HARD_CHUNKS_PER_FILE

        # Coverage matches DocIndex's shared tokenize()-based formula unchanged.
        q_tokens = set(tokenize(query))
        chunk_by_id = {c.id: c for c in fts_idx.chunks}
        for hit in fts_res:
            chunk = chunk_by_id[hit["id"]]
            expected_cov = len(q_tokens & set(tokenize(f"{chunk.header}\n{chunk.text}"))) / len(
                q_tokens
            )
            assert hit["coverage"] == pytest.approx(expected_cov)
        bm25_cov_by_id = {hit["id"]: hit["coverage"] for hit in bm25_res}
        for hit in fts_res:
            if hit["id"] in bm25_cov_by_id:
                assert hit["coverage"] == pytest.approx(bm25_cov_by_id[hit["id"]])


def test_pretok_matches_bm25_returned_set_on_every_golden_question(tmp_path: Path) -> None:
    golden = load_golden(GOLDEN_PATH)
    bm25_idx = DocIndex(CORPUS_DIR, retriever_factory=Bm25Retriever)
    bm25_idx.refresh()

    db_path = tmp_path / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_path, tmp_path, tokenizer="pretok") as retriever:
        fts_idx = DocIndex(CORPUS_DIR, retriever_factory=lambda: retriever)
        fts_idx.refresh()

        total_chunks = len(bm25_idx.chunks)
        assert total_chunks == len(fts_idx.chunks)

        bm25_ret = Bm25Retriever()
        bm25_ret.build(bm25_idx.chunks)

        for row in golden:
            q = str(row["q"])
            q_tokens = set(tokenize(q))

            bm25_ids = {r["id"] for r in bm25_idx.search(q, k=total_chunks)}
            fts_ids = {r["id"] for r in fts_idx.search(q, k=total_chunks)}
            assert fts_ids == bm25_ids, f"Returned chunk ID set mismatch for {row['id']}: {q!r}"

            bm25_indices = {idx for idx, _ in bm25_ret.rank(q, total_chunks)}
            fts_indices = {idx for idx, _ in retriever.rank(q, total_chunks)}
            assert fts_indices == bm25_indices, f"Rank index set mismatch for {row['id']}"

            # Double-check zero-overlap chunks are never returned.
            for idx in fts_indices:
                chunk = fts_idx.chunks[idx]
                chunk_tokens = set(tokenize(f"{chunk.header}\n{chunk.text}"))
                assert q_tokens & chunk_tokens, f"Chunk {chunk.id} shares no token with {q!r}"


@pytest.mark.parametrize("tokenizer", ["pretok", "porter"])
def test_golden_fixture_floors_both_variants(
    tmp_path: Path,
    tokenizer: Literal["pretok", "porter"],
) -> None:
    golden = load_golden(GOLDEN_PATH)
    db_path = tmp_path / tokenizer / "index.sqlite3"
    with Fts5Retriever(db_path, tmp_path, tokenizer=tokenizer) as retriever:
        index = DocIndex(CORPUS_DIR, retriever_factory=lambda: retriever)
        index.refresh()

        overall_report = evaluate(index.search, golden, ks=(1, 3, 5))
        table = format_metrics_table(overall_report, ks=(1, 3, 5))
        print(f"\n=== fts5 ({tokenizer}) ===\n" + table)
        assert overall_report["overall"]["n"] == 29
        overall_r3 = overall_report["overall"]["recall"][3]
        overall_mrr = overall_report["overall"]["mrr"]
        assert overall_r3 >= 0.862, (
            f"{tokenizer} overall R@3={overall_r3:.4f} < 0.862 (MRR={overall_mrr:.4f})"
        )

        floor_rows = [row for row in golden if str(row["type"]) in FLOOR_TYPES]
        floor_report = evaluate(index.search, floor_rows, ks=(1, 3, 5))
        assert floor_report["overall"]["n"] == 20
        lex_num_id_r3 = floor_report["overall"]["recall"][3]
        assert lex_num_id_r3 >= 0.90, f"{tokenizer} lex+num+id R@3={lex_num_id_r3:.4f} < 0.90"


@pytest.mark.parametrize("tokenizer", ["pretok", "porter"])
def test_adversarial_queries_never_raise_or_alter_result_set(
    tmp_path: Path,
    tokenizer: Literal["pretok", "porter"],
) -> None:
    with Fts5Retriever(None, tmp_path, tokenizer=tokenizer) as retriever:
        index = DocIndex(CORPUS_DIR, retriever_factory=lambda: retriever)
        index.refresh()
        k = len(index.chunks)

        base_query = "router paperclip"
        base_ids = [hit["id"] for hit in index.search(base_query, k=k)]
        assert base_ids

        adversarial_same_tokens = [
            '"router" "paperclip"',
            "router* paperclip*",
            "router:paperclip",
            "^router ^paperclip",
            "(router) ((paperclip))",
            "router\x00paperclip",
            "router AND OR paperclip",  # 'and'/'or' are stopwords in doc_chat.corpus
            '"router" * : ^ ( ) \x00 ' + ("paperclip " * 1400),  # > 10k chars
            '"unmatched quote router paperclip',
            'router "" "" paperclip',
        ]
        for adv in adversarial_same_tokens:
            hits = index.search(adv, k=k)
            assert [hit["id"] for hit in hits] == base_ids, f"Changed result for {adv[:40]!r}"

        # FTS5 boolean/proximity keywords are tokenized as plain literal terms, never operators.
        operator_queries = [
            "router NEAR paperclip",
            "router NOT paperclip",
            "router AND paperclip OR firmware",
            "NEAR(router paperclip, 5)",
        ]
        for op_query in operator_queries:
            plain_equivalent = " ".join(tokenize(op_query))
            op_ids = [hit["id"] for hit in index.search(op_query, k=k)]
            plain_ids = [hit["id"] for hit in index.search(plain_equivalent, k=k)]
            assert op_ids == plain_ids

        # > 64 unique tokens (> 10k chars total) never raises or hits SQLite limits.
        many_terms = " ".join(f"uniqueterm{i} " * 20 for i in range(120))
        assert len(many_terms) > 10_000
        assert index.search(many_terms + " router", k=k) == []

        # Syntax-only queries yield empty results cleanly.
        for syntax_only in ['""', "*", "()", ":", "^", "\x00", "AND OR", "   "]:
            assert index.search(syntax_only, k=k) == []


def test_incremental_sync_and_restart_zero_writes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(5):
        (docs / f"doc_{i}.md").write_text(
            f"# Title {i}\n\nDocument {i} discusses topic{i} and commonterm.\n",
            encoding="utf-8",
        )
    db_path = docs / ".rag_cache" / "fts5" / "index.sqlite3"

    with Fts5Retriever(db_path, docs, tokenizer="pretok") as retriever:
        assert retriever.status() == "fts5:file"
        index = DocIndex(docs, retriever_factory=lambda: retriever)
        index.refresh()
        assert retriever.last_sync_stats == SyncStats(inserted=5, deleted=0)

        assert retriever._store is not None
        conn = retriever._store._conn
        assert conn is not None

        # Unchanged refresh does 0 writes.
        trace_log: list[str] = []
        conn.set_trace_callback(trace_log.append)
        changes_before = conn.total_changes
        index.refresh()
        assert conn.total_changes == changes_before
        assert trace_log == []

        # Edit 1 of 5 files -> rewrites only that 1 file's rows.
        (docs / "doc_2.md").write_text(
            "# Title 2\n\nUpdated document 2 discusses brandnewkeyword.\n",
            encoding="utf-8",
        )
        changes_before_edit = conn.total_changes
        trace_log.clear()
        index.refresh()
        assert retriever.last_sync_stats == SyncStats(inserted=1, deleted=1)
        assert conn.total_changes > changes_before_edit
        assert sum(1 for s in trace_log if s.startswith("DELETE FROM chunks")) == 1
        assert sum(1 for s in trace_log if s.startswith("INSERT INTO chunks")) == 1
        assert sum(1 for s in trace_log if s.startswith("DELETE FROM fts")) == 1
        assert sum(1 for s in trace_log if s.startswith("INSERT INTO fts")) == 1
        conn.set_trace_callback(None)

        assert [h["source"] for h in index.search("brandnewkeyword", k=4)] == ["doc_2.md"]
        assert index.search("topic2", k=4) == []
        assert [h["source"] for h in index.search("topic0", k=4)] == ["doc_0.md"]

    # Restart with unchanged files -> 0 writes on reopen + refresh.
    with Fts5Retriever(db_path, docs, tokenizer="pretok") as reopened:
        assert reopened.status() == "fts5:file"
        assert reopened._store is not None
        reopen_conn = reopened._store._conn
        assert reopen_conn is not None
        reopen_trace: list[str] = []
        reopen_conn.set_trace_callback(reopen_trace.append)
        changes_on_reopen = reopen_conn.total_changes

        index2 = DocIndex(docs, retriever_factory=lambda: reopened)
        index2.refresh()

        assert reopened.last_sync_stats == SyncStats(inserted=0, deleted=0)
        assert reopen_conn.total_changes == changes_on_reopen
        assert all(stmt.startswith("SELECT ") for stmt in reopen_trace)
        reopen_conn.set_trace_callback(None)
        assert [h["source"] for h in index2.search("brandnewkeyword", k=4)] == ["doc_2.md"]


def test_probe_failure_serves_bm25_with_single_log_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(fts5_mod, "fts5_available", lambda _tok: (False, "no such module: fts5"))

    chunks = [
        _make_chunk(0, "a.md", "a.md", "alpha bravo charlie"),
        _make_chunk(1, "b.md", "b.md", "delta echo foxtrot"),
    ]
    bm25_ref = Bm25Retriever()
    bm25_ref.build(chunks)
    expected = bm25_ref.rank("bravo", 2)

    db_path = tmp_path / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_path, tmp_path, tokenizer="pretok") as retriever:
        assert retriever.status() == "bm25 (fts5 unavailable: no such module: fts5)"
        retriever.build(chunks)
        assert retriever.rank("bravo", 2) == expected
        # Subsequent build/rank calls keep serving BM25 without extra log lines.
        retriever.build(chunks)
        assert retriever.rank("bravo", 2) == expected
        assert not db_path.exists()

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert err_lines == ["rag: fts5 unavailable (no such module: fts5); serving BM25"]


def test_store_open_failure_both_file_and_memory_falls_back_to_bm25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _broken_store(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("sqlite init failed")

    monkeypatch.setattr(fts5_mod, "Fts5Store", _broken_store)
    chunks = [_make_chunk(0, "a.md", "a.md", "gamma delta epsilon")]

    with Fts5Retriever(tmp_path / "index.sqlite3", tmp_path) as retriever:
        assert retriever.status().startswith("bm25 (fts5 unavailable:")
        retriever.build(chunks)
        assert len(retriever.rank("delta", 2)) == 1

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert err_lines[0].startswith("rag: fts5 unavailable (")
    assert err_lines[0].endswith("); serving BM25")


def test_runtime_sqlite_error_in_rank_and_build_rebuilds_bm25(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks = [
        _make_chunk(0, "a.md", "a.md", "apollo launch countdown"),
        _make_chunk(1, "b.md", "b.md", "voyager deep space probe"),
    ]
    bm25_ref = Bm25Retriever()
    bm25_ref.build(chunks)

    # 1. Runtime error during rank() rebuilds Bm25Retriever from the last chunk list.
    with Fts5Retriever(None, tmp_path) as retriever:
        retriever.build(chunks)
        assert retriever._store is not None

        def _boom_query(*_args: object, **_kwargs: object) -> list[tuple[str, float]]:
            raise sqlite3.DatabaseError("database disk image is malformed")

        retriever._store.query = _boom_query  # type: ignore[method-assign]
        hits = retriever.rank("apollo", 2)
        assert hits == bm25_ref.rank("apollo", 2)
        assert retriever.status() == (
            "bm25 (fts5 unavailable: DatabaseError: database disk image is malformed)"
        )
        # Subsequent rank stays on BM25 without logging again.
        assert retriever.rank("voyager", 2) == bm25_ref.rank("voyager", 2)

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert err_lines == [
        "rag: fts5 unavailable (DatabaseError: database disk image is malformed); serving BM25"
    ]

    # 2. Runtime error during build() rebuilds Bm25Retriever from the incoming chunk list.
    with Fts5Retriever(None, tmp_path) as retriever2:
        assert retriever2._store is not None

        def _boom_sync(*_args: object, **_kwargs: object) -> SyncStats:
            raise sqlite3.OperationalError("disk I/O error")

        retriever2._store.sync_lazy = _boom_sync  # type: ignore[method-assign]
        retriever2.build(chunks)
        assert retriever2.rank("apollo", 2) == bm25_ref.rank("apollo", 2)
        assert retriever2.status() == "bm25 (fts5 unavailable: OperationalError: disk I/O error)"

    err_lines2 = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert err_lines2 == ["rag: fts5 unavailable (OperationalError: disk I/O error); serving BM25"]


def test_docindex_over_readonly_rag_corpus_and_chmod_readonly_dir(tmp_path: Path) -> None:
    # 1. Over committed tests/fixtures/rag_corpus without modifying it.
    before_entries = sorted(p.name for p in CORPUS_DIR.iterdir())
    with Fts5Retriever(None, CORPUS_DIR) as retriever:
        assert retriever.status() == "fts5:memory"
        index = DocIndex(CORPUS_DIR, retriever_factory=lambda: retriever)
        index.refresh()
        assert index.failures() == []
        hits = index.search("router", k=4)
        assert hits
    assert sorted(p.name for p in CORPUS_DIR.iterdir()) == before_entries

    # 2. Over a chmod 0555 read-only docs directory where disk DB creation fails -> fts5:memory.
    ro_docs = tmp_path / "ro_docs"
    ro_docs.mkdir()
    shutil.copy2(CORPUS_DIR / "router_reset.md", ro_docs / "router_reset.md")
    ro_docs.chmod(0o555)
    try:
        db_path = ro_docs / ".rag_cache" / "fts5" / "index.sqlite3"
        with Fts5Retriever(db_path, ro_docs) as ro_retriever:
            assert ro_retriever.status() == "fts5:memory"
            ro_index = DocIndex(ro_docs, retriever_factory=lambda: ro_retriever)
            ro_index.refresh()
            assert ro_index.search("router", k=2)
    finally:
        ro_docs.chmod(0o755)


def test_build_tokenizes_only_new_or_changed_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _make_chunk(i, f"doc_{i}.md", f"doc_{i}.md > Section {i}", f"alpha topic{i} body")
        for i in range(5)
    ]

    # 1. Pretok mode: cold build tokenizes all 5; unchanged restart tokenizes 0; edit 1 -> 1.
    pretok_calls: list[str] = []
    tokenize_calls: list[str] = []
    orig_pretok_row = fts5_mod._pretok_row
    orig_tokenize = fts5_mod.tokenize

    def _counting_pretok(chunk: Chunk) -> tuple[str, str, str, str]:
        pretok_calls.append(chunk.id)
        return orig_pretok_row(chunk)

    def _counting_tokenize(text: str) -> list[str]:
        tokenize_calls.append(text)
        return orig_tokenize(text)

    monkeypatch.setattr(fts5_mod, "_pretok_row", _counting_pretok)
    monkeypatch.setattr(fts5_mod, "tokenize", _counting_tokenize)

    db_pretok = tmp_path / "pretok" / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_pretok, tmp_path / "pretok", tokenizer="pretok") as retriever:
        retriever.build(chunks)
        assert retriever.last_sync_stats == SyncStats(inserted=5, deleted=0)
        assert sorted(pretok_calls) == sorted(c.id for c in chunks)
        assert len(tokenize_calls) == 5

        pretok_calls.clear()
        tokenize_calls.clear()
        retriever.build(chunks)
        assert retriever.last_sync_stats == SyncStats(inserted=0, deleted=0)
        assert pretok_calls == []
        assert tokenize_calls == []

    # Reopen from disk with unchanged chunks: 0 row/tokenize calls.
    with Fts5Retriever(db_pretok, tmp_path / "pretok", tokenizer="pretok") as reopened:
        reopened.build(chunks)
        assert reopened.last_sync_stats == SyncStats(inserted=0, deleted=0)
        assert pretok_calls == []
        assert tokenize_calls == []

        # Edit 1 of 5 chunks: only that chunk is tokenized.
        edited = list(chunks)
        edited[2] = _make_chunk(
            2,
            "doc_2.md",
            "doc_2.md > Section 2",
            "alpha brandnewpretok body",
        )
        reopened.build(edited)
        assert reopened.last_sync_stats == SyncStats(inserted=1, deleted=1)
        assert pretok_calls == ["doc_2.md#2"]
        assert tokenize_calls == ["alpha brandnewpretok body"]
        assert [idx for idx, _ in reopened.rank("brandnewpretok", 5)] == [2]
        assert reopened.rank("topic2", 5) == []

    # 2. Porter mode: unchanged restart calls _porter_row 0 times; edit 1 -> 1.
    porter_calls: list[str] = []
    orig_porter_row = fts5_mod._porter_row

    def _counting_porter(chunk: Chunk) -> tuple[str, str, str, str]:
        porter_calls.append(chunk.id)
        return orig_porter_row(chunk)

    monkeypatch.setattr(fts5_mod, "_porter_row", _counting_porter)

    db_porter = tmp_path / "porter" / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Retriever(db_porter, tmp_path / "porter", tokenizer="porter") as p_ret:
        p_ret.build(chunks)
        assert p_ret.last_sync_stats == SyncStats(inserted=5, deleted=0)
        assert sorted(porter_calls) == sorted(c.id for c in chunks)

    porter_calls.clear()
    with Fts5Retriever(db_porter, tmp_path / "porter", tokenizer="porter") as p_reopened:
        p_reopened.build(chunks)
        assert p_reopened.last_sync_stats == SyncStats(inserted=0, deleted=0)
        assert porter_calls == []

        edited_porter = list(chunks)
        edited_porter[1] = _make_chunk(
            1,
            "doc_1.md",
            "doc_1.md > Section 1",
            "alpha brandnewporter body",
        )
        p_reopened.build(edited_porter)
        assert p_reopened.last_sync_stats == SyncStats(inserted=1, deleted=1)
        assert porter_calls == ["doc_1.md#1"]
        assert [idx for idx, _ in p_reopened.rank("brandnewporter", 5)] == [1]


def test_chunk_key_parity_with_pretok_and_porter_rows(tmp_path: Path) -> None:
    samples = [
        _make_chunk(0, "a.md", "a.md > Intro", "hello world"),
        _make_chunk(1, "b.md", "b.md", ""),
        _make_chunk(2, "c.md", "c.md > Quotes 'n' Unicode", "café naïve don't\nline two"),
    ]
    for chunk in samples:
        expected_key = _chunk_key(chunk.header, chunk.text)
        assert fts5_mod._pretok_row(chunk)[1] == expected_key
        assert fts5_mod._porter_row(chunk)[1] == expected_key

    for tok in ("pretok", "porter"):
        mode: Literal["pretok", "porter"] = tok
        with Fts5Retriever(None, tmp_path, tokenizer=mode) as retriever:
            retriever.build(samples)
            assert retriever._store is not None
            assert retriever._store._conn is not None
            stored = dict(
                retriever._store._conn.execute("SELECT chunk_id, key FROM chunks").fetchall()
            )
            for chunk in samples:
                assert stored[chunk.id] == _chunk_key(chunk.header, chunk.text)


def test_lock_contention_uses_memory_fallback_preserves_inode_and_logs_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    db_path = docs / ".rag_cache" / "fts5" / "index.sqlite3"
    chunks = [
        _make_chunk(0, "a.md", "a.md", "unique_secret_text_alpha calibration"),
        _make_chunk(1, "b.md", "b.md", "unique_secret_text_beta maintenance"),
    ]

    with Fts5Retriever(db_path, docs) as seed:
        seed.build(chunks)
    before_stat = db_path.stat()

    holder = sqlite3.connect(db_path, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        with Fts5Retriever(
            db_path,
            docs,
            _busy_timeout=0.05,
            _lock_timeout=0.05,
            _retry_delay=0.02,
        ) as retriever:
            assert retriever.status() == "fts5:memory"
            retriever.build(chunks)
            assert [idx for idx, _ in retriever.rank("calibration", 2)] == [0]
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    after_stat = db_path.stat()
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_dev == before_stat.st_dev

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert "OperationalError" in err_lines[0]
    assert str(docs) not in err_lines[0]
    assert str(db_path) not in err_lines[0]
    assert "unique_secret_text" not in err_lines[0]


def test_disk_store_unavailable_logs_one_stderr_line_without_path_or_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ro_docs = tmp_path / "ro_docs_secret_folder"
    ro_docs.mkdir()
    (ro_docs / "guide.md").write_text(
        "# Guide\n\nConfidential doc body token_xyz_999.\n",
        encoding="utf-8",
    )
    ro_docs.chmod(0o555)
    try:
        db_path = ro_docs / ".rag_cache" / "fts5" / "index.sqlite3"
        with Fts5Retriever(db_path, ro_docs) as retriever:
            assert retriever.status() == "fts5:memory"
            idx = DocIndex(ro_docs, retriever_factory=lambda: retriever)
            idx.refresh()
            assert idx.search("Confidential", k=2)
    finally:
        ro_docs.chmod(0o755)

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert "PermissionError" in err_lines[0]
    assert str(ro_docs) not in err_lines[0]
    assert str(db_path) not in err_lines[0]
    assert "ro_docs_secret_folder" not in err_lines[0]
    assert "token_xyz_999" not in err_lines[0]


def test_transient_sqlite_errors_in_build_and_rank_do_not_stick_to_bm25(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks = [
        _make_chunk(0, "a.md", "a.md", "apollo launch countdown"),
        _make_chunk(1, "b.md", "b.md", "voyager deep space probe"),
    ]
    db_path = tmp_path / ".rag_cache" / "fts5" / "index.sqlite3"

    with Fts5Retriever(db_path, tmp_path, _retry_delay=0.01) as retriever:
        retriever.build(chunks)
        assert retriever.status() == "fts5:file"
        assert retriever._store is not None
        orig_query = retriever._store.query
        orig_sync = retriever._store.sync_lazy

        # 1. Transient BUSY error in rank() succeeds on retry and stays on FTS5.
        query_calls = 0

        def _busy_once_query(
            match: str,
            limit: int | None = None,
            *,
            weights: tuple[float, ...] | None = None,
        ) -> list[tuple[str, float]]:
            nonlocal query_calls
            query_calls += 1
            if query_calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return orig_query(match, limit=limit, weights=weights)

        retriever._store.query = _busy_once_query  # type: ignore[method-assign]
        assert [idx for idx, _ in retriever.rank("apollo", 2)] == [0]
        assert query_calls == 2
        assert retriever.status() == "fts5:file"
        assert retriever._bm25 is None

        # 2. Repeated transient error in rank() serves query without permanently switching to BM25.
        def _busy_always_query(*_args: object, **_kwargs: object) -> list[tuple[str, float]]:
            raise sqlite3.OperationalError("database table is locked")

        retriever._store.query = _busy_always_query  # type: ignore[method-assign]
        assert [idx for idx, _ in retriever.rank("voyager", 2)] == [1]
        assert retriever.status() == "fts5:file"
        assert retriever._bm25 is None

        # Restore query: still using FTS5 store!
        retriever._store.query = orig_query  # type: ignore[method-assign]
        assert [idx for idx, _ in retriever.rank("apollo", 2)] == [0]
        assert retriever.status() == "fts5:file"

        # 3. Transient disk-full error in build() retries once; repeats stay on FTS5.
        sync_calls = 0

        def _disk_full_once(
            keyed: object,
            make_row: object,
        ) -> SyncStats:
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 1:
                raise sqlite3.OperationalError("database or disk is full")
            return orig_sync(keyed, make_row)  # type: ignore[arg-type]

        retriever._store.sync_lazy = _disk_full_once  # type: ignore[method-assign]
        updated = chunks + [_make_chunk(2, "c.md", "c.md", "cassini saturn orbiter")]
        retriever.build(updated)
        assert sync_calls == 2
        assert retriever.status() == "fts5:file"
        assert retriever._bm25 is None
        assert [idx for idx, _ in retriever.rank("cassini", 2)] == [2]

        def _busy_always_sync(*_args: object, **_kwargs: object) -> SyncStats:
            raise sqlite3.OperationalError("database is locked")

        retriever._store.sync_lazy = _busy_always_sync  # type: ignore[method-assign]
        updated2 = updated + [_make_chunk(3, "d.md", "d.md", "juno jupiter mission")]
        retriever.build(updated2)
        assert retriever.status() == "fts5:file"
        assert retriever._bm25 is None

        # Restore sync_lazy: pending sync completes on FTS5.
        retriever._store.sync_lazy = orig_sync  # type: ignore[method-assign]
        assert [idx for idx, _ in retriever.rank("juno", 2)] == [3]
        assert retriever.status() == "fts5:file"
        assert retriever._bm25 is None

    assert [line for line in capsys.readouterr().err.splitlines() if line.strip()] == []
