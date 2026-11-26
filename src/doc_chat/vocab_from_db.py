"""Speech vocabulary for --entities from an ingest DB, after ingest (ticket 219):
name -> category ("name" | "acronym" | "term" | "id") for
src/lan_speech/hotwords.py load_entities. Sources: chunk text, file names and
file-card titles (when the table exists)."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

from doc_chat.names import extract_names
from lan_speech.transcript_repair import COMMON_WORDS

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,10}\b")
_ID_RE = re.compile(r"\b[A-Za-z]{1,6} ?\d{1,5}(?:[A-Za-z]\d{0,5}|-\d{1,4})?\b")
_TERM_RE = re.compile(r"\b[a-z]{5,24}\b")
_BOOST = 100  # added to the score when a candidate appears in a file name/card title


def _tables(conn: sqlite3.Connection) -> set[str]:
    """The table names of a DB (the file_cards table may not exist)."""
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _boosted(cand: str, where: str) -> bool:
    """True when cand appears as a whole word in a file name or card title."""
    pat = r"(?<![0-9a-zA-Z])" + re.escape(cand.casefold()) + r"(?![0-9a-zA-Z])"
    return re.search(pat, where.casefold()) is not None


def vocab_from_db(db_path: Path | str, *, limit: int = 400) -> dict[str, str]:
    """The top `limit` vocabulary items of the corpus in db_path as
    name -> category, ranked by corpus count plus a file-name/card-title boost
    (ticket 219)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
        text = "\n".join(r[0] for r in conn.execute("SELECT text FROM chunk_data"))
        file_names = ""
        if "files" in tables:
            file_names = " ".join(r[0] for r in conn.execute("SELECT rel_key FROM files"))
        cards = ""
        if "file_cards" in tables:
            cards = " ".join(r[0] for r in conn.execute("SELECT card_text FROM file_cards"))
    finally:
        conn.close()
    return _rank(text, file_names + " " + cards, limit)


def _rank(text: str, where: str, limit: int) -> dict[str, str]:
    """The candidates as name -> category, top `limit` by (score desc, name).
    Categories in precedence order: name (extract_names), acronym (ALL-CAPS),
    id (mixed letters and digits), term (lowercase, not an English common
    word). Score = corpus count + _BOOST when found in a file name/card title."""
    common = {w.lower() for w in COMMON_WORDS}
    cands: dict[str, tuple[str, int]] = {}

    def add(name: str, category: str, count: int) -> None:
        if name.casefold() in common or name in cands:
            return
        cands[name] = (category, count + (_BOOST if _boosted(name, where) else 0))

    for name, count in extract_names(text, min_count=2, max_words=5).items():
        add(name, "name", count)
    for token, count in Counter(_ACRONYM_RE.findall(text)).items():
        if count >= 2:
            add(token, "acronym", count)
    for token, count in Counter(_ID_RE.findall(text)).items():
        if count >= 2:
            add(token, "id", count)
    for token, count in Counter(_TERM_RE.findall(text.lower())).items():
        if count >= 3:
            add(token, "term", count)
    ranked = sorted(cands.items(), key=lambda kv: (-kv[1][1], kv[0]))
    return {name: cat for name, (cat, _score) in ranked[:limit]}
