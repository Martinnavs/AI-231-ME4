"""scripts/me4_mockday.py: the hand-made files are readable by the real indexer."""

import importlib.util
import sys
import zipfile
from pathlib import Path

from voice_agent.rag.docindex import DocIndex

_spec = importlib.util.spec_from_file_location(
    "me4_mockday", Path(__file__).resolve().parents[1] / "scripts" / "me4_mockday.py"
)
assert _spec is not None and _spec.loader is not None
mockday = importlib.util.module_from_spec(_spec)
sys.modules["me4_mockday"] = mockday
_spec.loader.exec_module(mockday)


def test_zip_holds_the_picks_and_the_five_handmade_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "gov").mkdir(parents=True)
    (raw / "gov" / "a.txt").write_text("alpha")
    out = tmp_path / "m.zip"
    names = mockday.build_mock_zip(out, raw, ["gov/a.txt"])
    assert len(names) == 6 and names[0] == "ipo-2026/gov/a.txt"
    assert sorted(zipfile.ZipFile(out).namelist()) == sorted(names)


def test_every_handmade_file_is_indexed_ok(tmp_path: Path) -> None:
    for name, data in mockday.handmade_files().items():
        target = tmp_path / "docs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    index = DocIndex(tmp_path / "docs")
    index.refresh()
    assert sorted(r.status for r in index.records()) == ["ok"] * 5
    assert index.failures() == []
    hits = index.search("When does Project Heron ship", 4)
    assert any("28 June" in h["text"] for h in hits)
