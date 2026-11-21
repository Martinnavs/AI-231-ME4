import zipfile

import pytest

from voice_agent.rag import zipingest


def _info(raw: bytes, flag_bits: int = 0) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("x")
    info.filename = raw.decode("cp437")
    info.flag_bits = flag_bits
    return info


def test_utf8_flag_name_is_kept() -> None:
    info = zipfile.ZipInfo("café.txt")
    info.flag_bits = 0x800
    assert zipingest.decode_member_name(info) == "café.txt"


def test_unflagged_utf8_bytes_are_decoded_as_utf8() -> None:
    info = _info("café.txt".encode())
    assert zipingest.decode_member_name(info) == "café.txt"


def test_unflagged_non_utf8_bytes_stay_cp437() -> None:
    info = _info("café.txt".encode("cp437"))
    assert zipingest.decode_member_name(info) == "café.txt"


def test_backslashes_become_slashes() -> None:
    assert zipingest.decode_member_name(_info(b"a\\b.txt")) == "a/b.txt"


@pytest.mark.parametrize(
    ("name", "junk"),
    [("__MACOSX/a/b.txt", True), ("docs/._b.txt", True), ("docs/b.txt", False)],
)
def test_is_junk_member(name: str, junk: bool) -> None:
    assert zipingest.is_junk_member(name) is junk
