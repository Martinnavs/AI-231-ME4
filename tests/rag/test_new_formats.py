"""Offline unit tests for TSV and plain-text format indexing (ticket 60)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.rag import DocIndex


def test_tsv_indexed_and_searchable(tmp_path: Path) -> None:
    tsv = tmp_path / "data.tsv"
    tsv.write_text("Name\tPrice\tStock\nWidget\t100\t50\nGadget\t250\t30\n", encoding="utf-8")
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["data.tsv"]
    assert report.failures == []
    hits = idx.search("widget price 100", k=2)
    assert hits
    assert hits[0]["source"] == "data.tsv"


def test_plain_text_formats_indexed(tmp_path: Path) -> None:
    body = "Configuration for the solar panel array includes 42 modules rated at 350 watts each.\n"
    for _ext, name in (".log", "sys.log"), (".ini", "app.ini"), (".toml", "cfg.toml"):
        (tmp_path / name).write_text(body, encoding="utf-8")
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert sorted(report.added) == ["app.ini", "cfg.toml", "sys.log"]
    assert report.failures == []
    hits = idx.search("42 modules rated at 350 watts", k=3)
    assert len(hits) == 3
    assert {h["source"] for h in hits} == {"sys.log", "app.ini", "cfg.toml"}


@pytest.mark.parametrize("ext", [".yaml", ".cfg", ".conf"])
def test_yaml_cfg_conf_indexed(tmp_path: Path, ext: str) -> None:
    stem = "test" + ext
    body = "Database host is db.example.com on port 5432 for production use.\n"
    (tmp_path / stem).write_text(body, encoding="utf-8")
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == [stem]
    assert report.failures == []
    hits = idx.search("db.example.com port 5432", k=2)
    assert hits
    assert hits[0]["source"] == stem
