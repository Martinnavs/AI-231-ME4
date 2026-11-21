"""Safe extraction of a dropped .zip into the corpus folder (archive limits, not OOXML limits)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

MAX_TOTAL_BYTES = 1024**3
MAX_MEMBER_BYTES = 200 * 1024 * 1024
MAX_MEMBERS = 5000
MAX_RATIO = 100.0
RATIO_MIN_BYTES = 1024 * 1024
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar")
_UTF8_FLAG = 0x800
_ENCRYPTED_FLAG = 0x1
LEDGER_NAME = "zips.json"


def decode_member_name(info: zipfile.ZipInfo) -> str:
    """UTF-8 flag, else strict UTF-8 over the raw bytes, else cp437; backslashes become '/'."""
    name = info.filename
    if not info.flag_bits & _UTF8_FLAG:
        try:
            name = name.encode("cp437").decode("utf-8")
        except UnicodeError:
            name = info.filename
    return name.replace("\\", "/")


def is_junk_member(name: str) -> bool:
    """True for macOS resource-fork debris: anything under __MACOSX or a '._*' file name."""
    parts = PurePosixPath(name).parts
    return "__MACOSX" in parts or (bool(parts) and parts[-1].startswith("._"))


def _refusal(name: str, why: str) -> str:
    return f"{ascii(name[:80])}: {why}"


def validate_members(
    infos: list[zipfile.ZipInfo], dest: Path
) -> tuple[list[tuple[zipfile.ZipInfo, Path]], list[str], int]:
    """Split members into (accepted with target path, refusal reasons, junk skipped)."""
    if len(infos) > MAX_MEMBERS:
        return [], [f"archive: {len(infos)} members, limit {MAX_MEMBERS}"], 0
    if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        return [], [f"archive: declared size over {MAX_TOTAL_BYTES} bytes"], 0
    root = dest.resolve()
    accepted: list[tuple[zipfile.ZipInfo, Path]] = []
    refused: list[str] = []
    seen: set[str] = set()
    junk = 0
    for info in infos:
        name = decode_member_name(info)
        if is_junk_member(name):
            junk += 1
            continue
        if info.is_dir():
            continue
        target = (root / name).resolve()
        if stat.S_ISLNK(info.external_attr >> 16):
            why = "symlink"
        elif info.flag_bits & _ENCRYPTED_FLAG:
            why = "encrypted"
        elif name.lower().endswith(ARCHIVE_SUFFIXES):
            why = "nested archive"
        elif target == root or not target.is_relative_to(root):
            why = "path escapes the folder"
        elif name.casefold() in seen:
            why = "duplicate name"
        elif info.file_size > MAX_MEMBER_BYTES:
            why = "member too large"
        elif info.file_size >= RATIO_MIN_BYTES and info.file_size > MAX_RATIO * info.compress_size:
            why = "compression ratio too high"
        else:
            seen.add(name.casefold())
            accepted.append((info, target))
            continue
        refused.append(_refusal(name, why))
    return accepted, refused, junk


def _copy_bounded(src: Any, dst: Any, limit: int) -> int:
    written = 0
    while chunk := src.read(65536):
        written += len(chunk)
        if written > limit:
            raise ValueError("size over limit")
        dst.write(chunk)
    return written


def extract_zip(zip_path: Path, corpus: Path) -> dict[str, Any]:
    """Extract into <corpus>/<stem>/ via a hidden staging folder; return the manifest."""
    stem = zip_path.stem.lstrip(".") or "archive"
    dest = corpus / stem
    staging = corpus / f".{stem}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    manifest: dict[str, Any] = {"zip": zip_path.name, "dest": stem, "extracted": [], "refused": []}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            accepted, refused, junk = validate_members(zf.infolist(), staging)
            manifest["refused"] = refused
            manifest["junk_skipped"] = junk
            total = 0
            for info, target in accepted:
                limit = min(MAX_MEMBER_BYTES, MAX_TOTAL_BYTES - total)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, target.open("wb") as out:
                        total += _copy_bounded(src, out, limit)
                except (OSError, ValueError, EOFError, zlib.error, zipfile.BadZipFile) as exc:
                    target.unlink(missing_ok=True)
                    refused.append(_refusal(info.filename, f"not extracted ({type(exc).__name__})"))
                    continue
                manifest["extracted"].append(target.relative_to(staging.resolve()).as_posix())
    except zipfile.BadZipFile:
        manifest["refused"] = ["archive: not a valid zip file"]
        shutil.rmtree(staging, ignore_errors=True)
        return manifest
    if dest.exists():
        old = corpus / f".{stem}.old-{os.getpid()}"
        os.rename(dest, old)
        os.rename(staging, dest)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.rename(staging, dest)
    return manifest


def ingest_zips(corpus: Path, extra: list[Path] | None = None) -> list[tuple[str, str]]:
    """Extract each new or changed zip (corpus top level plus extra); return (zip name, why)."""
    ledger_path = corpus / ".rag_cache" / LEDGER_NAME
    ledger: dict[str, Any] = {}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    problems: list[tuple[str, str]] = []
    changed = False
    dropped = sorted(p for p in corpus.iterdir() if p.suffix.lower() == ".zip" and p.is_file())
    for zip_path in [*dropped, *(extra or [])]:
        st = zip_path.stat()
        key = str(zip_path.resolve())
        sig = f"{st.st_size}:{st.st_mtime_ns}"
        entry = ledger.get(key)
        if entry is None or entry["sig"] != sig:
            manifest = extract_zip(zip_path, corpus)
            entry = {"sig": sig, "refused": manifest["refused"]}
            ledger[key] = entry
            changed = True
        problems.extend((zip_path.name, why) for why in entry["refused"])
    if changed:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger), encoding="utf-8")
        os.replace(tmp, ledger_path)
    return problems
