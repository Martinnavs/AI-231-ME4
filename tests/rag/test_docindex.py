"""Offline unit tests for voice_agent.rag.docindex."""

from __future__ import annotations

import copy
import io
import os
import shutil
import signal
import time
import warnings
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from voice_agent.rag import Chunk, DocIndex, FileRecord, RefreshReport, Retriever
from voice_agent.rag import docindex as docindex_mod


def _pad_sentences(prefix: str, count: int = 12) -> str:
    return " ".join(
        f"{prefix} sentence number {i} contains enough words for realistic chunking."
        for i in range(count)
    )


def test_loaders_metadata_chunk_fields_and_dates(tmp_path: Path) -> None:
    md_file = tmp_path / "guide.md"
    md_file.write_text(
        "Introductory paragraph before headings that is long enough to keep as a chunk.\n\n"
        "# Setup\n\n"
        "Short.\n\n"
        "## Install\n\n"
        "Install the package on 2026-03-15 or May 4, 2026 following standard steps.\n"
        "```python\n"
        "# not a markdown heading inside code\n"
        "x = 1\n"
        "```\n\n"
        "# Usage\n\n"
        "Usage instructions with enough characters to clear the minimum sixty-character floor.\n",
        encoding="utf-8",
    )

    txt_file = tmp_path / "notes.txt"
    txt_bytes = (
        b"\xef\xbb\xbf"
        b"BOM notes created on November 18, 2026 and updated 2026-11-19 with \xff detail."
    )
    txt_file.write_bytes(txt_bytes)

    csv_file = tmp_path / "inventory.csv"
    rows = ["item,price,updated"]
    for i in range(2100):
        rows.append(f"widget_{i},{100 + i},2026-01-01")
    csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert isinstance(report, RefreshReport)
    assert sorted(report.added) == ["guide.md", "inventory.csv", "notes.txt"]
    assert report.failures == []

    records = {rec.name: rec for rec in idx.records()}
    assert set(records) == {"guide.md", "notes.txt", "inventory.csv"}

    md_rec = records["guide.md"]
    assert isinstance(md_rec, FileRecord)
    assert md_rec.path == str(md_file)
    assert md_rec.kind == "md"
    assert md_rec.size == md_file.stat().st_size
    assert md_rec.mtime == pytest.approx(md_file.stat().st_mtime)
    assert md_rec.ingested_at
    assert len(md_rec.sha256) == 64
    assert md_rec.dates_in_text == ["2026-03-15", "2026-05-04"]
    assert md_rec.status == "ok"
    assert md_rec.error is None

    txt_rec = records["notes.txt"]
    assert txt_rec.kind == "txt"
    assert txt_rec.dates_in_text == ["2026-11-18", "2026-11-19"]
    assert txt_rec.status == "ok"

    md_chunks = [c for c in idx.chunks if c.source == "guide.md"]
    assert [c.position for c in md_chunks] == list(range(len(md_chunks)))
    assert [c.id for c in md_chunks] == [f"guide.md#{i}" for i in range(len(md_chunks))]
    sections = [c.section for c in md_chunks]
    assert "" in sections
    assert "Setup" in sections
    assert "Setup > Install" in sections
    assert "Usage" in sections
    setup_chunk = next(c for c in md_chunks if c.section == "Setup")
    assert setup_chunk.text == "Short."

    install_chunk = next(c for c in md_chunks if c.section == "Setup > Install")
    assert install_chunk.header == "guide.md > Setup > Install"
    assert not install_chunk.text.startswith("guide.md")
    assert "# not a markdown heading" in install_chunk.text
    assert install_chunk.page is None
    assert len(install_chunk.file_date) == 10 and install_chunk.file_date[4] == "-"
    assert install_chunk.file_sha256 == md_rec.sha256
    assert len(install_chunk.chunk_sha256) == 64

    txt_chunk = next(c for c in idx.chunks if c.source == "notes.txt")
    assert txt_chunk.header == "notes.txt"
    assert not txt_chunk.text.startswith("\ufeff")
    assert txt_chunk.text.startswith("BOM notes")
    assert "\ufffd" in txt_chunk.text

    csv_text = "\n".join(c.text for c in idx.chunks if c.source == "inventory.csv")
    assert "item: widget_0; price: 100; updated: 2026-01-01" in csv_text
    assert "item: widget_1999; price: 2099; updated: 2026-01-01" in csv_text
    assert "widget_2000" not in csv_text


def test_chunk_overlap_short_section_kept_and_hash_stability(tmp_path: Path) -> None:
    long_body = _pad_sentences("Overlap verification", count=30)
    assert len(long_body) > 1500

    (tmp_path / "long.md").write_text(
        "# Heading Stub\n\nTiny.\n\n# Long Section\n\n" + long_body + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dup_a.md").write_text(
        "# Shared\n\n"
        "Identical paragraph shared between two files to test chunk content hashing stability.\n",
        encoding="utf-8",
    )
    (tmp_path / "dup_b.md").write_text(
        "# Shared\n\n"
        "Identical paragraph shared between two files to test chunk content hashing stability.\n\n"
        "# Extra\n\n" + _pad_sentences("Extra second section", count=4),
        encoding="utf-8",
    )

    idx1 = DocIndex(tmp_path)
    idx1.refresh()

    long_chunks = [c for c in idx1.chunks if c.source == "long.md"]
    assert len(long_chunks) >= 3
    assert long_chunks[0].section == "Heading Stub"
    assert long_chunks[0].text == "Tiny."
    long_section_chunks = [c for c in long_chunks if c.section == "Long Section"]
    assert len(long_section_chunks) >= 2
    assert all(c.text.strip() for c in idx1.chunks)

    # Verify non-empty overlap between consecutive chunks of the long section.
    first_tail_words = long_section_chunks[0].text.split()[-8:]
    assert " ".join(first_tail_words) in long_section_chunks[1].text

    dup_a = next(c for c in idx1.chunks if c.source == "dup_a.md")
    dup_b = next(c for c in idx1.chunks if c.source == "dup_b.md" and c.position == 0)
    assert dup_a.chunk_sha256 == dup_b.chunk_sha256
    assert dup_a.file_sha256 != dup_b.file_sha256

    snapshot_1 = [(c.id, c.chunk_sha256, c.file_sha256) for c in idx1.chunks]
    idx1.refresh()
    snapshot_2 = [(c.id, c.chunk_sha256, c.file_sha256) for c in idx1.chunks]
    assert snapshot_1 == snapshot_2

    idx2 = DocIndex(tmp_path)
    idx2.refresh()
    snapshot_3 = [(c.id, c.chunk_sha256, c.file_sha256) for c in idx2.chunks]
    assert snapshot_1 == snapshot_3


def test_change_detection_and_no_reparse_on_unchanged_or_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha = tmp_path / "alpha.md"
    beta = tmp_path / "beta.txt"
    alpha.write_text("# Alpha\n\n" + _pad_sentences("Alpha initial content", 4), encoding="utf-8")
    beta.write_text(_pad_sentences("Beta initial content", 4), encoding="utf-8")

    hash_calls = 0
    load_calls = 0
    chunk_calls = 0
    orig_hash = docindex_mod.hash_bytes
    orig_load = docindex_mod.decode_and_load
    orig_chunk = docindex_mod.chunk_text

    def count_hash(data: bytes) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return orig_hash(data)

    def count_load(raw_bytes: bytes, kind: str) -> tuple[str, str, str | None]:
        nonlocal load_calls
        load_calls += 1
        return orig_load(raw_bytes, kind)

    def count_chunk(*args: object, **kwargs: object) -> list[Chunk]:
        nonlocal chunk_calls
        chunk_calls += 1
        return orig_chunk(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(docindex_mod, "hash_bytes", count_hash)
    monkeypatch.setattr(docindex_mod, "decode_and_load", count_load)
    monkeypatch.setattr(docindex_mod, "chunk_text", count_chunk)

    idx = DocIndex(tmp_path)
    r1 = idx.refresh()
    assert sorted(r1.added) == ["alpha.md", "beta.txt"]
    assert r1.unchanged == 0
    assert load_calls == 2
    assert chunk_calls == 2

    # Unchanged folder: zero hashing, loading, or chunking.
    hash_calls = load_calls = chunk_calls = 0
    r2 = idx.refresh()
    assert r2.added == []
    assert r2.changed == []
    assert r2.removed == []
    assert r2.unchanged == 2
    assert hash_calls == 0
    assert load_calls == 0
    assert chunk_calls == 0

    # Touch alpha.md with identical contents: re-hashes once, does not re-parse or re-chunk.
    stat = alpha.stat()
    os.utime(alpha, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    hash_calls = load_calls = chunk_calls = 0
    r3 = idx.refresh()
    assert r3.changed == []
    assert r3.unchanged == 2
    assert hash_calls == 1
    assert load_calls == 0
    assert chunk_calls == 0

    # Edit alpha.md, delete beta.txt, add gamma.md.
    alpha.write_text("# Alpha\n\n" + _pad_sentences("Alpha updated zebra content", 4))
    beta.unlink()
    gamma = tmp_path / "gamma.md"
    gamma.write_text("# Gamma\n\n" + _pad_sentences("Gamma brand new content", 4))

    r4 = idx.refresh()
    assert r4.added == ["gamma.md"]
    assert r4.changed == ["alpha.md"]
    assert r4.removed == ["beta.txt"]
    assert r4.unchanged == 0
    assert any("zebra" in hit["text"] for hit in idx.search("zebra"))
    assert idx.search("Beta initial") == []


def test_failures_caps_missing_folder_symlinks_and_hidden_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_idx = DocIndex(tmp_path / "does_not_exist")
    missing_report = missing_idx.refresh()
    assert missing_report.added == []
    assert missing_report.failures == []
    assert missing_idx.search("anything") == []

    outside = tmp_path / "outside.txt"
    outside.write_text(_pad_sentences("Secret outside symlink target", 4), encoding="utf-8")
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "leak.txt").write_text(
        _pad_sentences("Outside directory leak", 4), encoding="utf-8"
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.txt").write_text(
        "Valid indexed document with enough characters for one valid searchable chunk.",
        encoding="utf-8",
    )
    (docs / "empty.txt").write_text("", encoding="utf-8")
    (docs / "garbage.txt").write_bytes(b"\xff\xfe" * 50)
    (docs / "archive.bin").write_text("binary placeholder", encoding="utf-8")
    (docs / "huge.txt").write_text("oversized " * 25, encoding="utf-8")
    (docs / ".hidden.md").write_text(_pad_sentences("Hidden dotfile", 4), encoding="utf-8")
    (docs / "~$temp.txt").write_text(_pad_sentences("Office temp file", 4), encoding="utf-8")
    (docs / ".~lock.doc.md#").write_text(_pad_sentences("LibreOffice lock", 4), encoding="utf-8")
    (docs / "escape.txt").symlink_to(outside)
    (docs / "escape_dir").symlink_to(outside_dir)

    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 150)

    idx = DocIndex(docs)
    report = idx.refresh()
    assert report.added == ["good.txt"]

    failures = {name: (status, err) for name, status, err in report.failures}
    assert failures["empty.txt"][0] == "empty"
    assert failures["garbage.txt"][0] == "unreadable"
    assert failures["archive.bin"][0] == "unsupported"
    assert failures["huge.txt"][0] == "too_large"
    assert idx.failures() == report.failures
    assert ".hidden.md" not in failures
    assert "~$temp.txt" not in failures
    assert ".~lock.doc.md#" not in failures
    assert "escape.txt" not in failures
    assert "escape_dir" not in failures

    all_text = "\n".join(c.text for c in idx.chunks)
    assert "outside symlink" not in all_text
    assert "Outside directory leak" not in all_text
    assert "Hidden dotfile" not in all_text
    assert "Office temp" not in all_text
    assert "LibreOffice lock" not in all_text
    assert idx.search("oversized") == []
    assert idx.search("binary placeholder") == []


def test_retriever_protocol_search_keys_and_coverage(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "# Alpha Topic\n\n"
        "Alpha document discusses apples bananas and cherries in detailed prose for testing.\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "# Beta Topic\n\n"
        "Beta document discusses apples dates and figs in detailed prose for testing.\n",
        encoding="utf-8",
    )

    class ReverseFakeRetriever:
        def __init__(self) -> None:
            self._chunks: Sequence[Chunk] = []

        def build(self, chunks: Sequence[Chunk]) -> None:
            self._chunks = list(chunks)

        def rank(self, query: str, k: int) -> list[tuple[int, float]]:
            del query
            # Rank chunk 1 ahead of chunk 0 regardless of lexical score.
            pairs = [(i, float(i + 1)) for i in range(len(self._chunks) - 1, -1, -1)]
            return pairs[:k]

    def factory() -> Retriever:
        return ReverseFakeRetriever()

    idx = DocIndex(tmp_path, retriever_factory=factory)
    idx.refresh()

    assert idx.search("") == []
    assert idx.search("   ") == []

    results = idx.search("what about apples bananas cherries dragonfruit", k=2)
    assert len(results) == 2
    assert [r["source"] for r in results] == ["b.md", "a.md"]
    assert [r["score"] for r in results] == [2.0, 1.0]

    expected_keys = {"source", "id", "section", "page", "file_date", "text", "score", "coverage"}
    for item in results:
        assert set(item.keys()) == expected_keys

    # Query content tokens: {appl, banana, cherri, dragonfruit} (4 unique after tokenize).
    # b.md has {appl} -> 1/4 = 0.25; a.md has {appl, banana, cherri} -> 3/4 = 0.75.
    assert results[0]["coverage"] == pytest.approx(0.25)
    assert results[1]["coverage"] == pytest.approx(0.75)


def test_synthetic_300_chunk_timing_bounds(tmp_path: Path) -> None:
    # 40 files * ~8 chunks each > 300 chunks.
    paragraph = _pad_sentences("Synthetic benchmark topic", count=20)
    for file_idx in range(40):
        sections = [
            f"## Section {sec_idx}\n\n{paragraph} keyword_{file_idx}_{sec_idx}.\n"
            for sec_idx in range(4)
        ]
        (tmp_path / f"doc_{file_idx:02d}.md").write_text("\n".join(sections), encoding="utf-8")

    idx = DocIndex(tmp_path)
    t0 = time.perf_counter()
    first = idx.refresh()
    first_elapsed = time.perf_counter() - t0

    assert len(first.added) == 40
    assert len(idx.chunks) >= 300
    assert first_elapsed < 10.0

    t1 = time.perf_counter()
    second = idx.refresh()
    hits = idx.search("keyword_17_2", k=4)
    second_plus_search = time.perf_counter() - t1

    assert second.unchanged == 40
    assert second_plus_search < 1.0
    assert hits and hits[0]["source"] == "doc_17.md"
    print(
        f"\n[timing] chunks={len(idx.chunks)} first_refresh={first_elapsed:.4f}s "
        f"unchanged_refresh_plus_search={second_plus_search:.4f}s"
    )


def test_header_weighting_ranks_topic_file_above_overview(tmp_path: Path) -> None:
    (tmp_path / "08-the-agent-loop.md").write_text(
        "# The Agent Loop\n\nThe agent loop runs one model call per step until no tools remain.\n",
        encoding="utf-8",
    )
    (tmp_path / "00-overview.md").write_text(
        "# Key Facts One\n\n"
        "This overview mentions an agent and a loop once while explaining config defaults.\n\n"
        "# Key Facts Two\n\n"
        "This overview mentions an agent and a loop once while explaining local secrets.\n\n"
        "# Key Facts Three\n\n"
        "This overview mentions an agent and a loop once while explaining model providers.\n",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path)
    idx.refresh()
    hits = idx.search("What does the agent loop do?", k=3)
    assert hits
    assert hits[0]["source"] == "08-the-agent-loop.md"


def test_hyphen_and_underscore_file_stem_searchable(tmp_path: Path) -> None:
    (tmp_path / "08-deploy_pipeline-guide.md").write_text(
        "Follow these instructions carefully to launch the service in production safely.\n",
        encoding="utf-8",
    )
    (tmp_path / "alpha.md").write_text(
        "Alpha document covers unrelated local database migrations and backups cleanly.\n",
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text(
        "Beta document covers unrelated audio framing and microphone capture timings.\n",
        encoding="utf-8",
    )

    assert docindex_mod._tokenize_header("08-deploy_pipeline-guide.md") == [
        "deploy",
        "pipeline",
        "guide",
    ]

    idx = DocIndex(tmp_path)
    idx.refresh()
    hits = idx.search("deploy pipeline", k=3)
    assert hits
    assert hits[0]["source"] == "08-deploy_pipeline-guide.md"
    assert idx.search("08", k=3) == []


def test_per_file_cap_limits_single_file_when_other_files_match(tmp_path: Path) -> None:
    dominant_sections = "\n\n".join(
        f"# Part {i}\n\n"
        f"Quantum computing uses quantum gates and quantum bits for quantum simulation {i}."
        for i in range(4)
    )
    (tmp_path / "dominant.md").write_text(dominant_sections + "\n", encoding="utf-8")
    (tmp_path / "secondary.md").write_text(
        "# Notes\n\nQuantum hardware requires cryogenic cooling to keep noise low in practice.\n",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path)
    idx.refresh()
    hits = idx.search("quantum", k=3)
    assert len(hits) == 3
    sources = [h["source"] for h in hits]
    assert sources.count("dominant.md") <= 2
    assert "secondary.md" in sources


def _make_simple_docx(document_xml: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def _force_out_of_range_mtime(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set an out-of-range mtime on ``path`` via utime or a Path.stat wrapper if FS clamps it."""
    try:
        os.utime(path, (0, 300_000_000_000))
        try:
            datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except (ValueError, OverflowError, OSError):
            return
    except (OSError, OverflowError):
        pass

    orig_stat = Path.stat

    class _FakeStat:
        def __init__(self, real: os.stat_result) -> None:
            self._real = real
            self.st_mtime_ns = 10**24
            self.st_mtime = 1e15

        def __getattr__(self, item: str) -> object:
            return getattr(self._real, item)

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        real = orig_stat(self, *args, **kwargs)  # type: ignore[arg-type]
        if self == path:
            return _FakeStat(real)
        return real

    monkeypatch.setattr(Path, "stat", fake_stat)


def test_refresh_per_file_guard_records_exception_type_and_caches_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "good.txt"
    good.write_text(
        "Healthy document about photovoltaic panel voltage tolerances in summer.",
        encoding="utf-8",
    )
    flaky = tmp_path / "flaky.md"
    flaky.write_text(
        "# Flaky\n\n" + _pad_sentences("Initial healthy flaky text", 4), encoding="utf-8"
    )

    idx = DocIndex(tmp_path)
    r1 = idx.refresh()
    assert sorted(r1.added) == ["flaky.md", "good.txt"]
    assert idx.failures() == []

    flaky.write_text(
        "# Flaky\n\nTRIGGER_ERROR " + _pad_sentences("Updated flaky content", 4),
        encoding="utf-8",
    )
    orig_decode = docindex_mod.decode_and_load
    guarded_calls = 0

    def failing_decode(raw_bytes: bytes, kind: str) -> tuple[str, str, str | None]:
        nonlocal guarded_calls
        if b"TRIGGER_ERROR" in raw_bytes:
            guarded_calls += 1
            raise RuntimeError("secret /home/alice/private_path leaked")
        return orig_decode(raw_bytes, kind)

    monkeypatch.setattr(docindex_mod, "decode_and_load", failing_decode)

    r2 = idx.refresh()
    assert r2.removed == ["flaky.md"]
    assert idx.failures() == [("flaky.md", "unreadable", "could not be read (RuntimeError)")]
    assert r2.failures == idx.failures()
    err_reason = idx.failures()[0][2]
    assert "RuntimeError" in err_reason
    assert "secret" not in err_reason
    assert "/home" not in err_reason
    assert all(c.source != "flaky.md" for c in idx.chunks)
    assert idx.search("photovoltaic panel voltage")
    assert guarded_calls == 1

    # Next refresh with unchanged stat reports the cached failure without re-reading.
    guarded_calls = 0
    r3 = idx.refresh()
    assert guarded_calls == 0
    assert r3.failures == [("flaky.md", "unreadable", "could not be read (RuntimeError)")]
    assert idx.failures() == r3.failures

    # BaseException (KeyboardInterrupt) must not be swallowed.
    kb_file = tmp_path / "kb.txt"
    kb_file.write_text("Keyboard interrupt propagation check document body.", encoding="utf-8")

    def raise_kb(raw_bytes: bytes, kind: str) -> tuple[str, str, str | None]:
        if b"Keyboard interrupt" in raw_bytes:
            raise KeyboardInterrupt
        return orig_decode(raw_bytes, kind)

    monkeypatch.setattr(docindex_mod, "decode_and_load", raise_kb)
    with pytest.raises(KeyboardInterrupt):
        idx.refresh()


def test_out_of_range_mtime_falls_back_without_dropping_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    future = tmp_path / "future.txt"
    future.write_text(
        "Future mtime document about zirconium reactor cooling procedure details.",
        encoding="utf-8",
    )
    _force_out_of_range_mtime(future, monkeypatch)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["future.txt"]
    assert idx.failures() == []
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert rec.mtime == 0.0
    assert idx.chunks[0].file_date == ""
    hits = idx.search("zirconium reactor cooling", k=2)
    assert hits and hits[0]["source"] == "future.txt"


def test_deep_folder_chain_is_capped_and_named(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text(
        "Root level document with enough words to index properly.", encoding="utf-8"
    )
    cur = tmp_path
    for level in range(1, 26):
        cur = cur / f"d{level:02d}"
        cur.mkdir()
        if level == 20:
            (cur / "depth20.txt").write_text(
                "Document containing albatross inside the maximum allowed folder depth.",
                encoding="utf-8",
            )
    (cur / "depth25.txt").write_text(
        "Document containing platypus beyond the maximum allowed folder depth.",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path)
    report = idx.refresh()

    expected_dir = "/".join(f"d{i:02d}" for i in range(1, 22)) + "/"
    expected_depth20 = "/".join(f"d{i:02d}" for i in range(1, 21)) + "/depth20.txt"
    assert sorted(report.added) == [expected_depth20, "root.txt"]
    assert idx.failures() == [
        (expected_dir, "unreadable", "folder nested deeper than 20 levels, not scanned")
    ]
    assert report.failures == idx.failures()
    dir_rec = next(r for r in idx.records() if r.name == expected_dir)
    assert dir_rec.kind == "dir"
    assert dir_rec.status == "unreadable"
    assert idx.search("platypus") == []
    assert idx.search("albatross")


def test_1100_deep_folder_chain_does_not_hit_recursion_error(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text(
        "Root file beside a very deep folder chain with enough words.", encoding="utf-8"
    )
    cur = tmp_path
    try:
        for _ in range(1100):
            cur = cur / "a"
            cur.mkdir()
    except OSError as exc:
        pytest.skip(f"filesystem cannot create 1100-deep folder chain: {exc}")

    idx = DocIndex(tmp_path)
    t0 = time.perf_counter()
    report = idx.refresh()
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0
    expected_dir = "/".join(["a"] * 21) + "/"
    assert report.added == ["root.txt"]
    assert idx.failures() == [
        (expected_dir, "unreadable", "folder nested deeper than 20 levels, not scanned")
    ]


def test_heading_regex_bomb_is_linear_and_body_searchable(tmp_path: Path) -> None:
    (tmp_path / "bomb.md").write_text(
        "# a" + (" " * 9000) + "x\n\n"
        "Normal body discussing xylophone calibration metrics in clear detail.\n",
        encoding="utf-8",
    )
    (tmp_path / "h250.md").write_text(
        "# a" + (" " * 246) + "x\n\n"
        "Body under a two hundred fifty character heading with enough words.\n",
        encoding="utf-8",
    )
    (tmp_path / "edge.md").write_text(
        "# ###\n\n"
        "Body under empty hashes line with plenty of characters to pass the sixty char floor.\n\n"
        "# C#\n\n"
        "Body under C sharp heading with plenty of characters to pass the sixty character floor.\n",
        encoding="utf-8",
    )

    def _on_alarm(signum: int, frame: object) -> None:
        raise TimeoutError("refresh() exceeded 2s regex timeout guard")

    prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(2)
    try:
        idx = DocIndex(tmp_path)
        t0 = time.perf_counter()
        idx.refresh()
        elapsed = time.perf_counter() - t0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    assert elapsed < 1.0
    hits = idx.search("xylophone calibration", k=2)
    assert hits and hits[0]["source"] == "bomb.md"

    bomb_chunks = [c for c in idx.chunks if c.source == "bomb.md"]
    assert bomb_chunks and all(c.section == "" for c in bomb_chunks)

    h250_chunks = [c for c in idx.chunks if c.source == "h250.md"]
    assert h250_chunks and h250_chunks[0].section.startswith("a")

    edge_chunks = [c for c in idx.chunks if c.source == "edge.md"]
    assert [c.section for c in edge_chunks] == ["", "C"]


def test_heading_size_caps_for_md_and_nested_sections(tmp_path: Path) -> None:
    huge_heading_line = "# " + " ".join(["word"] * 8000)
    heading_290 = "# " + ("w" * 288)
    assert len(heading_290) == 290
    para = _pad_sentences("Repeated paragraph for chunk count scaling check", count=11)
    body_200_chunks = "\n\n".join(f"{para} item_{i}." for i in range(200))
    (tmp_path / "big.md").write_text(
        f"{huge_heading_line}\n\n{heading_290}\n\n{body_200_chunks}\n",
        encoding="utf-8",
    )

    nested_body = _pad_sentences("Nested long heading section body text", count=3)
    (tmp_path / "nested.md").write_text(
        f"# {'a' * 150}\n\n## {'b' * 150}\n\n### {'c' * 150}\n\n{nested_body}\n",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path)
    idx.refresh()

    big_chunks = [c for c in idx.chunks if c.source == "big.md"]
    assert len(big_chunks) >= 200
    capped_heading_chunks = [c for c in big_chunks if c.section]
    assert capped_heading_chunks
    assert all(c.section == "w" * 120 for c in capped_heading_chunks)
    assert all(c.header == f"big.md > {'w' * 120}" for c in capped_heading_chunks)

    nested_chunks = [c for c in idx.chunks if c.source == "nested.md"]
    assert nested_chunks
    for c in nested_chunks:
        assert len(c.section) <= 300
        assert len(c.header) <= 300 + len("nested.md") + 3
    expected_nested = f"{'a' * 120} > {'b' * 120} > {'c' * 120}"[:300]
    assert nested_chunks[0].section == expected_nested


def test_refresh_survives_all_hostile_inputs_alongside_real_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_src = Path(__file__).resolve().parents[1] / "fixtures" / "rag_corpus"
    folder = tmp_path / "corpus"
    shutil.copytree(corpus_src, folder, ignore=shutil.ignore_patterns(".rag_cache"))

    # 1. Deep docx (3000 nested tables with unique attributes).
    deep_open = "".join(
        f'<w:tbl w:id="t_{i}"><w:tr w:r="r_{i}"><w:tc w:c="c_{i}">' for i in range(3000)
    )
    deep_close = "</w:tc></w:tr></w:tbl>" * 3000
    deep_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{deep_open}<w:p><w:r><w:t>Deep</w:t></w:r></w:p>{deep_close}</w:body>"
        "</w:document>"
    ).encode()
    (folder / "deep.docx").write_bytes(_make_simple_docx(deep_xml))

    # 2. 25-deep folder chain.
    cur = folder
    for level in range(1, 26):
        cur = cur / f"chain_{level:02d}"
        cur.mkdir()
    (cur / "unreachable.txt").write_text("Should not be scanned at depth 25.", encoding="utf-8")
    expected_deep_dir = "/".join(f"chain_{i:02d}" for i in range(1, 22)) + "/"

    # 3. Overlapped zip docx.
    ov_buf = io.BytesIO()
    ov_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>Overlapped docx paragraph text.</w:t></w:r></w:p></w:body>"
        b"</w:document>"
    )
    with zipfile.ZipFile(ov_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
        zf.writestr("word/document.xml", ov_xml)
        orig_info = zf.getinfo("word/document.xml")
        dup_info = copy.copy(orig_info)
        dup_info.filename = "word/overlap.xml"
        zf.filelist.append(dup_info)
    (folder / "overlapped.docx").write_bytes(ov_buf.getvalue())

    # 4. Duplicate-member zip docx.
    dup_buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(dup_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
            zf.writestr("word/document.xml", ov_xml)
            zf.writestr("word/document.xml", ov_xml)
    (folder / "duplicate.docx").write_bytes(dup_buf.getvalue())

    # 5. Regex-bomb markdown.
    (folder / "regex_bomb.md").write_text(
        "# a" + (" " * 9000) + "x\n\n"
        "Body about telemetry calibration sentinel beacon inside regex bomb file.\n",
        encoding="utf-8",
    )

    # 6. Future out-of-range mtime file.
    future = folder / "future_mtime.txt"
    future.write_text(
        "Future mtime file discussing molybdenum alloy tensile rating parameters.",
        encoding="utf-8",
    )
    _force_out_of_range_mtime(future, monkeypatch)

    def _on_alarm(signum: int, frame: object) -> None:
        raise TimeoutError("combined refresh() exceeded 5s timeout")

    prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(5)
    try:
        idx = DocIndex(folder)
        t0 = time.perf_counter()
        report = idx.refresh()
        elapsed = time.perf_counter() - t0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    assert elapsed < 5.0
    assert report.failures == idx.failures()
    failed_names = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failed_names) == {
        "deep.docx",
        "overlapped.docx",
        "duplicate.docx",
        expected_deep_dir,
    }
    assert all(status == "unreadable" for status, _ in failed_names.values())

    hits = idx.search("renovation budget 450,000 pesos", k=3)
    assert any(h["source"] == "budget_2026.md" for h in hits)

    future_hits = idx.search("molybdenum alloy tensile", k=2)
    assert future_hits and future_hits[0]["source"] == "future_mtime.txt"
    assert future_hits[0]["file_date"] == ""


def test_short_files_short_sections_and_heading_only_empty_status(tmp_path: Path) -> None:
    short_txt = tmp_path / "pin.txt"
    short_txt.write_text("Door pin code is 4918.", encoding="utf-8")
    assert len("Door pin code is 4918.") == 22

    normal_md = tmp_path / "directory.md"
    normal_md.write_text(
        "# Office Directory\n\n"
        "General reception desk hours run from 08:00 to 17:00 Monday through Friday.\n\n"
        "## Phone\n\n"
        "Call 555-0123 now.\n",
        encoding="utf-8",
    )

    heading_only = tmp_path / "headings.md"
    heading_only.write_text("# Only Title\n\n## Empty Subsection\n", encoding="utf-8")

    shrink_txt = tmp_path / "shrink.txt"
    shrink_txt.write_text(
        "Initial long document discussing pelican migration paths across coastal lagoons.",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert sorted(report.added) == ["directory.md", "pin.txt", "shrink.txt"]
    assert report.failures == [("headings.md", "empty", "no text to index")]
    assert idx.failures() == report.failures

    records = {r.name: r for r in idx.records()}
    assert records["pin.txt"].status == "ok"
    assert records["headings.md"].status == "empty"
    assert records["headings.md"].error == "no text to index"

    pin_hits = idx.search("door pin code 4918", k=2)
    assert pin_hits and pin_hits[0]["source"] == "pin.txt"

    phone_hits = idx.search("phone number", k=2)
    assert phone_hits and phone_hits[0]["source"] == "directory.md"
    assert phone_hits[0]["section"] == "Office Directory > Phone"
    assert "555-0123" in phone_hits[0]["text"]

    # Shrinking a file below 60 chars keeps it ok and searchable.
    shrink_txt.write_text("Pelican tag 77.", encoding="utf-8")
    st = shrink_txt.stat()
    os.utime(shrink_txt, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    r2 = idx.refresh()
    assert r2.changed == ["shrink.txt"]
    assert idx.failures() == [("headings.md", "empty", "no text to index")]
    shrink_hits = idx.search("pelican tag 77", k=2)
    assert shrink_hits and shrink_hits[0]["source"] == "shrink.txt"

    # Shrinking an ok file to heading-only removes its chunks and names it as empty.
    shrink_txt.write_text("   \n", encoding="utf-8")
    st2 = shrink_txt.stat()
    os.utime(shrink_txt, ns=(st2.st_atime_ns, st2.st_mtime_ns + 2_000_000_000))
    r3 = idx.refresh()
    assert r3.removed == ["shrink.txt"]
    assert ("shrink.txt", "empty", "file is empty") in idx.failures()
    assert idx.search("pelican tag 77", k=2) == []


def test_relative_path_identity_for_same_named_files_in_subfolders(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text(
        "# Alpha\n\n"
        "Lantern battery replacement checklist for north boathouse.\n\n"
        "## Part Two\n\n"
        "Lantern spare bulbs are stored on the north shelf.\n\n"
        "## Part Three\n\n"
        "Lantern oil reserve sits beside the north generator.\n",
        encoding="utf-8",
    )
    (tmp_path / "b" / "notes.md").write_text(
        "# Beta\n\nLantern inspection schedule for south dock.\n",
        encoding="utf-8",
    )
    (tmp_path / "a" / "bad.md").write_text("# Empty Heading Only\n", encoding="utf-8")

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert sorted(report.added) == ["a/notes.md", "b/notes.md"]
    assert idx.failures() == [("a/bad.md", "empty", "no text to index")]
    assert report.failures == idx.failures()

    records = {r.name: r for r in idx.records()}
    assert set(records) == {"a/notes.md", "b/notes.md", "a/bad.md"}

    a_chunks = [c for c in idx.chunks if c.source == "a/notes.md"]
    b_chunks = [c for c in idx.chunks if c.source == "b/notes.md"]
    assert [c.id for c in a_chunks] == ["a/notes.md#0", "a/notes.md#1", "a/notes.md#2"]
    assert [c.id for c in b_chunks] == ["b/notes.md#0"]
    assert a_chunks[0].header == "a/notes.md > Alpha"
    assert b_chunks[0].header == "b/notes.md > Beta"

    # Both subfolder files match "lantern", and MAX_CHUNKS_PER_FILE=2 applies per relative path.
    hits = idx.search("lantern", k=3)
    sources = [h["source"] for h in hits]
    assert "a/notes.md" in sources
    assert "b/notes.md" in sources
    assert sources.count("a/notes.md") <= 2

    # _tokenize_header still uses Path(...).stem, so "notes" matches a/notes.md and b/notes.md.
    stem_hits = idx.search("notes", k=4)
    assert {h["source"] for h in stem_hits} == {"a/notes.md", "b/notes.md"}


def test_same_size_preserved_mtime_change_detected_via_ctime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "code.txt"
    initial = b"alpha_123456"
    updated = b"zebra_654321"
    assert len(initial) == len(updated) == 12

    target.write_bytes(initial)
    orig_stat = target.stat()

    idx = DocIndex(tmp_path)
    r1 = idx.refresh()
    assert r1.added == ["code.txt"]
    assert idx.search("alpha_123456")

    # Wait briefly so filesystem ctime_ns advances, rewrite same-length content, restore mtime.
    time.sleep(0.02)
    target.write_bytes(updated)
    os.utime(target, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

    r2 = idx.refresh()
    assert r2.changed == ["code.txt"]
    assert idx.search("zebra_654321")
    assert idx.search("alpha_123456") == []

    # A ctime-only change with identical bytes hashes once and reuses prev.chunks.
    load_calls = 0
    chunk_calls = 0
    orig_load = docindex_mod.decode_and_load
    orig_chunk = docindex_mod.chunk_text

    def count_load(raw_bytes: bytes, kind: str) -> tuple[str, str, str | None]:
        nonlocal load_calls
        load_calls += 1
        return orig_load(raw_bytes, kind)

    def count_chunk(*args: object, **kwargs: object) -> list[Chunk]:
        nonlocal chunk_calls
        chunk_calls += 1
        return orig_chunk(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(docindex_mod, "decode_and_load", count_load)
    monkeypatch.setattr(docindex_mod, "chunk_text", count_chunk)

    time.sleep(0.02)
    target.write_bytes(updated)
    os.utime(target, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
    r3 = idx.refresh()
    assert r3.unchanged == 1
    assert r3.changed == []
    assert load_calls == 0
    assert chunk_calls == 0


def test_docindex_retriever_property_exposes_active_and_replaces_on_refresh(tmp_path: Path) -> None:
    built: list[docindex_mod.Bm25Retriever] = []

    def factory() -> docindex_mod.Bm25Retriever:
        r = docindex_mod.Bm25Retriever()
        built.append(r)
        return r

    idx = DocIndex(tmp_path, retriever_factory=factory)
    assert len(built) == 1
    first = idx.retriever
    assert first is built[0]
    with pytest.raises(AttributeError):
        idx.retriever = docindex_mod.Bm25Retriever()  # type: ignore[misc]

    # Empty refresh leaves content unchanged, so retriever stays the same object.
    idx.refresh()
    assert idx.retriever is first

    # Refresh that adds content builds and exposes a new Retriever instance.
    (tmp_path / "notes.md").write_text(
        "# Notes\n\nSolar panels supply the microgrid in Cebu.\n", encoding="utf-8"
    )
    idx.refresh()
    assert len(built) == 2
    assert idx.retriever is built[1]
    assert idx.retriever is not first
