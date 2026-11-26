"""Vocabulary for the speech server from an ingest DB (ticket 219)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from doc_chat.vocab_from_db import vocab_from_db
from voice_agent.rag.ingest import ingest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_entities.py"

TEXT = (
    "Cornelius Suttree grew hydroponics in the east bay. "
    "GSA issued the W25Q16 form for hydroponics. "
    "GSA checked the hydroponics shed with Cornelius Suttree. "
    "GSA reissued W25Q16. Cornelius Suttree filed it."
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    src = tmp_path / "docs"
    src.mkdir()
    (src / "Cornelius Suttree.md").write_text(TEXT + "\n", encoding="utf-8")
    (src / "rates.csv").write_text(
        "id,note\n1,Cornelius Suttree filed the rate change\n", encoding="utf-8"
    )
    out = tmp_path / "docs.db"
    ingest(src, out)
    return out


@pytest.fixture()
def me(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("make_entities", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "make_entities", mod)
    spec.loader.exec_module(mod)
    return mod


def test_vocab_from_db_categories(db: Path) -> None:
    d = vocab_from_db(db)
    assert d["Cornelius Suttree"] == "name"
    assert d["GSA"] == "acronym"
    assert d["W25Q16"] == "id"
    assert d["hydroponics"] == "term"
    assert not any(k.casefold() == "the" for k in d)


def test_vocab_from_db_limit_and_rank(db: Path, tmp_path: Path) -> None:
    full = vocab_from_db(db)
    assert vocab_from_db(db, limit=2) == dict(list(full.items())[:2])
    src = tmp_path / "guide"
    src.mkdir()
    (src / "hydroponics_guide.md").write_text(
        "hydroponics is calm. aquaculture is loud. " * 3 + "\n", encoding="utf-8"
    )
    out = tmp_path / "guide.db"
    ingest(src, out)
    d = vocab_from_db(out)
    assert list(d).index("hydroponics") < list(d).index("aquaculture")


def test_make_entities_db_mode(
    me, db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "entities.json"
    monkeypatch.setattr(sys, "argv", ["make_entities", "--db", str(db), "-o", str(out)])
    assert me.main() == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["Cornelius Suttree"] == "name"
    assert got["GSA"] == "acronym"
    assert got["W25Q16"] == "id"
    assert got["hydroponics"] == "term"
