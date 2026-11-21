"""Builders for hostile and awkward zip files; each returns the path it wrote."""

import stat
import zipfile
from pathlib import Path


def build_zip(path: Path, members: list[tuple[zipfile.ZipInfo | str, bytes]], **kw: int) -> Path:
    with zipfile.ZipFile(path, "w", **kw) as zf:  # type: ignore[arg-type]
        for name, data in members:
            zf.writestr(name, data)
    return path


def zip_slip(path: Path) -> Path:
    return build_zip(path, [("ok.txt", b"fine"), ("../../evil.txt", b"x"), ("/abs_evil.txt", b"x")])


def symlink(path: Path) -> Path:
    link = zipfile.ZipInfo("link.txt")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    return build_zip(path, [("ok.txt", b"fine"), (link, b"/etc/passwd")])


def ratio_bomb(path: Path) -> Path:
    return build_zip(
        path,
        [("ok.txt", b"fine"), ("bomb.txt", b"\0" * (20 * 1024 * 1024))],
        compression=zipfile.ZIP_DEFLATED,
    )


def many_members(path: Path, count: int = 10_000) -> Path:
    return build_zip(path, [(f"f{i}.txt", b"x") for i in range(count)])


def encrypted(path: Path) -> Path:
    """ok.txt plus locked.pdf whose general-purpose flag bit 0 (encrypted) is set by hand."""
    build_zip(path, [("ok.txt", b"fine"), ("locked.pdf", b"secret")])
    data = bytearray(path.read_bytes())
    for sig, name_at, flag_at in ((b"PK\x03\x04", 30, 6), (b"PK\x01\x02", 46, 8)):
        pos = data.find(sig)
        while pos != -1:
            if data[pos + name_at : pos + name_at + 10] == b"locked.pdf":
                data[pos + flag_at] |= 0x1
            pos = data.find(sig, pos + 4)
    path.write_bytes(bytes(data))
    return path


def raw_name(path: Path, raw: bytes) -> Path:
    """A zip whose one member name is the raw bytes ``raw`` with no UTF-8 flag."""
    placeholder = b"Q" * len(raw)
    build_zip(path, [(placeholder.decode() + ".txt", b"hello")])
    path.write_bytes(path.read_bytes().replace(placeholder, raw))
    return path


def case_duplicates(path: Path) -> Path:
    return build_zip(path, [("Report.txt", b"one"), ("report.TXT", b"two")])


def nested_zip(path: Path) -> Path:
    inner = build_zip(path.with_name("inner.zip"), [("a.txt", b"a")])
    return build_zip(path, [("ok.txt", b"fine"), ("inner.zip", inner.read_bytes())])


def macos_junk(path: Path) -> Path:
    return build_zip(path, [("d/ok.txt", b"fine"), ("__MACOSX/d/._ok.txt", b"x")])


def total_71_mb(path: Path) -> Path:
    block = bytes(range(256)) * 4096
    return build_zip(path, [(f"part{i:02d}.txt", block) for i in range(71)])
