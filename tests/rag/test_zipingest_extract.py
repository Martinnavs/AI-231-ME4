import zipfile
from pathlib import Path
from typing import Any

import pytest
import zip_fixtures

from voice_agent.rag import zipingest


def test_extract_writes_files_under_the_stem_folder(tmp_path: Path) -> None:
    zpath = zip_fixtures.macos_junk(tmp_path / "pack.zip")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = zipingest.extract_zip(zpath, corpus)
    assert (corpus / "pack" / "d" / "ok.txt").read_bytes() == b"fine"
    assert manifest["extracted"] == ["d/ok.txt"] and manifest["junk_skipped"] == 1
    assert [p.name for p in corpus.iterdir()] == ["pack"]


def test_extract_replaces_an_existing_folder(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "pack").mkdir(parents=True)
    (corpus / "pack" / "stale.txt").write_text("old")
    zipingest.extract_zip(zip_fixtures.macos_junk(tmp_path / "pack.zip"), corpus)
    assert not (corpus / "pack" / "stale.txt").exists()
    assert (corpus / "pack" / "d" / "ok.txt").exists()
    assert [p.name for p in corpus.iterdir()] == ["pack"]


def test_a_file_that_is_not_a_zip_is_reported(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    manifest = zipingest.extract_zip(tmp_path / "bad.zip", corpus)
    assert "not a valid zip" in manifest["refused"][0]
    assert list(corpus.iterdir()) == []


def test_actual_size_over_the_limit_aborts_that_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zpath = zip_fixtures.build_zip(tmp_path / "lie.zip", [("big.bin", b"x" * 2000)])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(zipingest, "MAX_MEMBER_BYTES", 1000)

    def lying_validate(infos: list[zipfile.ZipInfo], dest: Path) -> Any:
        return [(infos[0], dest.resolve() / "big.bin")], [], 0

    monkeypatch.setattr(zipingest, "validate_members", lying_validate)
    manifest = zipingest.extract_zip(zpath, corpus)
    assert "not extracted" in manifest["refused"][0]
    assert not (corpus / "lie" / "big.bin").exists()
