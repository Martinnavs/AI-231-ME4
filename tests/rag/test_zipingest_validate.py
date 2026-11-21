import stat
import zipfile
from pathlib import Path

import pytest

from voice_agent.rag import zipingest


def _info(name: str, size: int = 10, packed: int = 10, flag_bits: int = 0) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.file_size = size
    info.compress_size = packed
    info.flag_bits = flag_bits
    return info


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt"])
def test_path_escape_is_refused(tmp_path: Path, name: str) -> None:
    accepted, refused, _ = zipingest.validate_members([_info(name)], tmp_path)
    assert accepted == []
    assert "path escapes" in refused[0]


def test_good_member_maps_under_dest(tmp_path: Path) -> None:
    accepted, refused, _ = zipingest.validate_members([_info("a/b.txt")], tmp_path)
    assert refused == []
    assert accepted[0][1] == tmp_path.resolve() / "a" / "b.txt"


def test_symlink_is_refused(tmp_path: Path) -> None:
    info = _info("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert "symlink" in zipingest.validate_members([info], tmp_path)[1][0]


def test_encrypted_is_refused(tmp_path: Path) -> None:
    refused = zipingest.validate_members([_info("s.pdf", flag_bits=0x1)], tmp_path)[1]
    assert "encrypted" in refused[0]


def test_nested_archive_is_refused(tmp_path: Path) -> None:
    refused = zipingest.validate_members([_info("inner.ZIP")], tmp_path)[1]
    assert "nested archive" in refused[0]


def test_ratio_bomb_is_refused_but_small_repetitive_file_is_not(tmp_path: Path) -> None:
    bomb = _info("bomb.txt", size=50 * 1024 * 1024, packed=1000)
    small = _info("zeros.csv", size=50_000, packed=60)
    accepted, refused, _ = zipingest.validate_members([bomb, small], tmp_path)
    assert [i.filename for i, _ in accepted] == ["zeros.csv"]
    assert "ratio" in refused[0]


def test_member_over_limit_is_refused(tmp_path: Path) -> None:
    big = _info("big.bin", size=zipingest.MAX_MEMBER_BYTES + 1, packed=zipingest.MAX_MEMBER_BYTES)
    assert "too large" in zipingest.validate_members([big], tmp_path)[1][0]


def test_too_many_members_refuses_the_archive(tmp_path: Path) -> None:
    infos = [_info(f"f{i}.txt") for i in range(zipingest.MAX_MEMBERS + 1)]
    accepted, refused, _ = zipingest.validate_members(infos, tmp_path)
    assert accepted == [] and "members" in refused[0]


def test_case_duplicates_keep_the_first_and_junk_is_counted(tmp_path: Path) -> None:
    infos = [_info("A.txt"), _info("a.TXT"), _info("__MACOSX/._A.txt")]
    accepted, refused, junk = zipingest.validate_members(infos, tmp_path)
    assert [i.filename for i, _ in accepted] == ["A.txt"]
    assert "duplicate" in refused[0] and junk == 1
