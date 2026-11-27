"""Ticket 252: unsupported files are recorded once, generic junk is ignored.

Corpus-agnostic: all fixtures are synthetic and generated here.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from voice_agent.rag.ingest import ingest

MD_TEXT = (
    "# Plan\n\n"
    "The widget syncs on 2026-05-02 and keeps a backup copy.\n\n"
    "## Notes\n\n"
    "Restart the service before the quarterly review.\n"
)


def _write_folder(src: Path, names: dict[str, bytes | str]) -> None:
    src.mkdir(parents=True, exist_ok=True)
    for name, content in names.items():
        target = src / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)


def _read_files(db: Path) -> dict[str, dict]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute(
        "SELECT rel_key, status, error, kind, size, n_chunks, note FROM files"
    ).fetchall()
    conn.close()
    keys = ["rel_key", "status", "error", "kind", "size", "n_chunks", "note"]
    return {r[0]: dict(zip(keys, r, strict=False)) for r in rows}


def _read_chunks(db: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute("SELECT source, section, text FROM chunk_data ORDER BY ord").fetchall()
    conn.close()
    return [tuple(r) for r in rows]


def _chunk_count_for(db: Path, source: str) -> int:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    count = conn.execute("SELECT COUNT(*) FROM chunk_data WHERE source = ?", (source,)).fetchone()[
        0
    ]
    conn.close()
    return count


async def _docs_status(corpus: Path, db: Path) -> dict[str, Any]:
    from fastmcp import Client  # noqa: PLC0415

    from voice_agent.mcp_servers.docs import build_server  # noqa: PLC0415

    server = build_server(corpus, db=db)
    async with Client(server) as client:
        result = await client.call_tool("docs_status", {})
    return result.structured_content


def test_unsupported_recorded_junk_ignored(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_folder(
        src,
        {
            "c.md": MD_TEXT,
            "a.json": '{"alpha": 1, "beta": 2}\n',
            "b.odt": b"odt placeholder payload",
            ".DS_Store": b"\x00\x01\x02\x03",
        },
    )
    db = tmp_path / "ingest.db"
    ingest(src, db)

    files = _read_files(db)
    # .DS_Store is junk: not recorded at all. Exactly 2 unsupported rows.
    assert set(files) == {"c.md", "a.json", "b.odt"}

    for name, ext in (("a.json", ".json"), ("b.odt", ".odt")):
        row = files[name]
        assert row["status"] == "unsupported"
        assert ext in (row["error"] or "")
        assert row["note"] == ext
        assert int(row["n_chunks"]) == 0
        assert row["kind"] == ext.lstrip(".")

    assert files["c.md"]["status"] == "ok"
    assert int(files["c.md"]["n_chunks"]) >= 1
    # Unsupported files must not produce chunks.
    assert _chunk_count_for(db, "a.json") == 0
    assert _chunk_count_for(db, "b.odt") == 0
    # All chunks belong to the supported file.
    assert {chunk[0] for chunk in _read_chunks(db)} == {"c.md"}


def test_supported_chunks_identical_to_alone(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_folder(src, {"c.md": MD_TEXT})
    db = tmp_path / "ingest.db"

    ingest(src, db)
    chunks_before = _read_chunks(db)
    assert chunks_before

    _write_folder(src, {"a.json": "{}\n", "b.odt": b"odt placeholder payload"})
    ingest(src, db)
    chunks_after = _read_chunks(db)

    assert chunks_after == chunks_before
    files = _read_files(db)
    assert files["c.md"]["status"] == "ok"
    assert int(files["c.md"]["n_chunks"]) == len(chunks_before)


def test_reingest_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_folder(
        src,
        {
            "c.md": MD_TEXT,
            "a.json": '{"alpha": 1}\n',
            "b.odt": b"odt placeholder payload",
            ".DS_Store": b"\x00\x01",
        },
    )
    db = tmp_path / "ingest.db"

    ingest(src, db)
    files_first = _read_files(db)
    chunks_first = _read_chunks(db)

    ingest(src, db)
    files_second = _read_files(db)
    chunks_second = _read_chunks(db)

    assert files_second == files_first
    assert chunks_second == chunks_first
    assert files_second["a.json"]["status"] == "unsupported"
    assert files_second["b.odt"]["status"] == "unsupported"
    assert ".DS_Store" not in files_second


def test_generic_junk_names_ignored(tmp_path: Path) -> None:
    src = tmp_path / "src"
    macosx = "__MACOSX"
    _write_folder(
        src,
        {
            "d.md": "Quarterly notes.\n",
            "Thumbs.db": b"thumb placeholder",
            "desktop.ini": "[.ShellClassInfo]\n",
            f"{macosx}/cache.json": "{}\n",
            "empty.json": b"",
            "~$draft.docx": b"office temp placeholder",
            ".hidden.txt": "dotfile with a supported extension\n",
        },
    )
    db = tmp_path / "ingest.db"
    ingest(src, db)

    files = _read_files(db)
    # Every junk entry is ignored; only the real document remains.
    assert set(files) == {"d.md"}
    assert {chunk[0] for chunk in _read_chunks(db)} == {"d.md"}


def test_reader_exception_is_recorded(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_folder(
        src,
        {
            "good.txt": "hello world\n",
            "bad.pdf": b"\x00\x01\x02\x80\xff\xfe\x04\x05",
        },
    )
    db = tmp_path / "ingest.db"
    ingest(src, db)

    files = _read_files(db)
    assert "bad.pdf" in files
    row = files["bad.pdf"]
    assert row["status"] != "ok"
    assert row["error"]
    assert int(row["n_chunks"]) == 0

    from voice_agent.rag.dbindex import DbIndex  # noqa: PLC015

    index = DbIndex(db)
    failed = {name for name, _status, _reason in index.failures()}
    assert "bad.pdf" in failed
    assert "good.txt" not in failed


def test_docs_status_lists_unsupported(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_folder(
        src,
        {
            "c.md": MD_TEXT,
            "a.json": '{"alpha": 1}\n',
            "b.odt": b"odt placeholder payload",
            ".DS_Store": b"\x00\x01",
        },
    )
    db = tmp_path / "ingest.db"
    ingest(src, db)

    status = asyncio.run(_docs_status(src, db))
    by_name = {entry["name"]: entry for entry in status["files"]}
    assert set(by_name) == {"c.md", "a.json", "b.odt"}
    assert by_name["a.json"]["status"] == "unsupported"
    assert by_name["b.odt"]["status"] == "unsupported"
    assert ".DS_Store" not in by_name

    failures = {entry["name"]: entry for entry in status["failures"]}
    assert set(failures) == {"a.json", "b.odt"}
    assert failures["a.json"]["status"] == "unsupported"
    assert ".json" in failures["a.json"]["reason"]
    assert ".odt" in failures["b.odt"]["reason"]
    assert "c.md" not in failures
