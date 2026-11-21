import os
from pathlib import Path

import zip_fixtures

from voice_agent.rag import zipingest


def _extract(tmp_path: Path, build: object, *args: object) -> tuple[dict, Path]:  # type: ignore[type-arg]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    zpath = build(tmp_path / "t.zip", *args)  # type: ignore[operator]
    return zipingest.extract_zip(zpath, corpus), corpus / "t"


def _outside_is_clean(tmp_path: Path) -> bool:
    return sorted(os.listdir(tmp_path)) == ["corpus", "t.zip"]


def test_zip_slip_and_absolute_path(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.zip_slip)
    assert manifest["extracted"] == ["ok.txt"] and len(manifest["refused"]) == 2
    assert _outside_is_clean(tmp_path) and not Path("/abs_evil.txt").exists()


def test_symlink_member(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.symlink)
    assert manifest["extracted"] == ["ok.txt"] and "symlink" in manifest["refused"][0]
    assert not (dest / "link.txt").exists()


def test_ratio_bomb(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.ratio_bomb)
    assert "ratio" in manifest["refused"][0] and not (dest / "bomb.txt").exists()


def test_ten_thousand_members(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.many_members)
    assert manifest["extracted"] == [] and "members" in manifest["refused"][0]


def test_encrypted_member(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.encrypted)
    assert "encrypted" in manifest["refused"][0] and not (dest / "locked.pdf").exists()


def test_cp437_and_cp932_names_extract_inside_the_folder(tmp_path: Path) -> None:
    cp437, d437 = _extract(tmp_path, zip_fixtures.raw_name, "café".encode("cp437"))
    assert cp437["refused"] == [] and (d437 / "café.txt").exists()
    other = tmp_path / "o"
    other.mkdir()
    cp932, d932 = _extract(other, zip_fixtures.raw_name, "テスト".encode("cp932"))
    assert cp932["refused"] == [] and len(list(d932.iterdir())) == 1


def test_case_duplicate_and_nested_zip(tmp_path: Path) -> None:
    dup, ddup = _extract(tmp_path, zip_fixtures.case_duplicates)
    assert len(dup["extracted"]) == 1 and "duplicate" in dup["refused"][0]
    other = tmp_path / "o"
    other.mkdir()
    nested, dnest = _extract(other, zip_fixtures.nested_zip)
    assert nested["extracted"] == ["ok.txt"] and "nested archive" in nested["refused"][0]


def test_71_mb_archive_is_accepted(tmp_path: Path) -> None:
    manifest, dest = _extract(tmp_path, zip_fixtures.total_71_mb)
    assert manifest["refused"] == [] and len(manifest["extracted"]) == 71
