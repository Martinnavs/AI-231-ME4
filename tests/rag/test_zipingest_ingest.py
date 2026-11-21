import json
from pathlib import Path
from typing import Any

import pytest
import zip_fixtures

from voice_agent.rag import zipingest


def test_a_dropped_zip_is_extracted_and_refusals_are_reported(tmp_path: Path) -> None:
    zip_fixtures.symlink(tmp_path / "drop.zip")
    problems = zipingest.ingest_zips(tmp_path)
    assert (tmp_path / "drop" / "ok.txt").exists()
    assert len(problems) == 1 and problems[0][0] == "drop.zip" and "symlink" in problems[0][1]


def test_an_unchanged_zip_is_not_extracted_again_but_still_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_fixtures.symlink(tmp_path / "drop.zip")
    first = zipingest.ingest_zips(tmp_path)
    calls: list[Path] = []
    real = zipingest.extract_zip

    def counting(zip_path: Path, corpus: Path) -> dict[str, Any]:
        calls.append(zip_path)
        return real(zip_path, corpus)

    monkeypatch.setattr(zipingest, "extract_zip", counting)
    assert zipingest.ingest_zips(tmp_path) == first
    assert calls == []
    assert (tmp_path / ".rag_cache" / "zips.json").exists()


def test_an_upper_case_zip_extension_is_found(tmp_path: Path) -> None:
    zip_fixtures.macos_junk(tmp_path / "DROP.ZIP")
    assert zipingest.ingest_zips(tmp_path) == []
    assert (tmp_path / "DROP" / "d" / "ok.txt").exists()


def test_an_extra_zip_outside_the_corpus_is_ingested(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    extra = zip_fixtures.macos_junk(tmp_path / "away.zip")
    assert zipingest.ingest_zips(corpus, [extra]) == []
    assert (corpus / "away" / "d" / "ok.txt").exists()
    ledger = json.loads((corpus / ".rag_cache" / "zips.json").read_text())
    assert list(ledger.values())[0]["refused"] == []
