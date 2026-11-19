"""Offline unit tests for voice_agent.rag.canonical_cache."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

from voice_agent.rag.canonical_cache import MAX_CANONICAL_BYTES, CanonicalCache, cache_key


def _make_key(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_cache_key_matches_sha256_and_varies_with_every_field() -> None:
    src_sha = _make_key("doc-bytes")
    base = cache_key(src_sha, "pdf", "markitdown", "0.1.9", 1)
    assert len(base) == 64
    assert base == hashlib.sha256(f"{src_sha}|pdf|markitdown|0.1.9|1".encode()).hexdigest()
    assert cache_key(src_sha, "pdf", "markitdown", "0.1.9", "1") == base

    assert cache_key(_make_key("other"), "pdf", "markitdown", "0.1.9", 1) != base
    assert cache_key(src_sha, "docx", "markitdown", "0.1.9", 1) != base
    assert cache_key(src_sha, "pdf", "pypdf", "0.1.9", 1) != base
    assert cache_key(src_sha, "pdf", "markitdown", "0.1.10", 1) != base
    assert cache_key(src_sha, "pdf", "markitdown", "0.1.9", 2) != base


def test_does_not_import_docindex() -> None:
    assert "voice_agent.rag.docindex" not in sys.modules or (
        "docindex"
        not in Path(sys.modules["voice_agent.rag.canonical_cache"].__file__ or "").read_text(
            encoding="utf-8"
        )
    )


def test_hit_and_miss(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    cache_dir = root / ".rag_cache" / "canonical"
    cache = CanonicalCache(cache_dir, root)

    assert cache.enabled is True
    assert cache.directory == cache_dir
    assert cache.root == root

    key_a = cache_key(_make_key("a"), "pdf", "markitdown", "0.1.9", 1)
    key_b = cache_key(_make_key("b"), "txt", "identity", "1", 1)

    # Cold miss does not create the cache directory.
    assert cache.get(key_a) is None
    assert not cache_dir.exists()

    payload = "<!-- rag:converter=markitdown-0.1.9+pp1 -->\n# Heading\nCafé — 100%\n"
    assert cache.put(key_a, payload) is True
    assert (cache_dir / f"{key_a}.md").is_file()
    assert cache.get(key_a) == payload
    assert cache.get(key_b) is None

    # Invalid keys miss / fail safely without raising.
    for bad_key in ("", "short", "../escape", "a/b", "g" * 64, "a" * 63, "a" * 65):
        assert cache.get(bad_key) is None
        assert cache.put(bad_key, "hello") is False


def test_corrupt_utf8_returns_miss(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache_dir.mkdir(parents=True)
    cache = CanonicalCache(cache_dir, root)

    key = _make_key("corrupt")
    (cache_dir / f"{key}.md").write_bytes(b"valid prefix \xff\xfe\xfd invalid utf8")

    assert cache.get(key) is None


def test_oversize_returns_miss_and_put_rejects_oversize(tmp_path: Path) -> None:
    assert MAX_CANONICAL_BYTES == 20_000_000
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache_dir.mkdir(parents=True)
    cache = CanonicalCache(cache_dir, root)

    key_over = _make_key("oversize")
    (cache_dir / f"{key_over}.md").write_bytes(b"x" * (MAX_CANONICAL_BYTES + 1))
    assert cache.get(key_over) is None

    key_exact = _make_key("exact")
    (cache_dir / f"{key_exact}.md").write_bytes(b"y" * MAX_CANONICAL_BYTES)
    loaded = cache.get(key_exact)
    assert loaded is not None
    assert len(loaded) == MAX_CANONICAL_BYTES

    assert cache.put(_make_key("too_big_put"), "z" * (MAX_CANONICAL_BYTES + 1)) is False
    assert not (cache_dir / f"{_make_key('too_big_put')}.md").exists()


def test_symlinked_entry_and_directory_entry_return_miss(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache_dir.mkdir(parents=True)
    cache = CanonicalCache(cache_dir, root)

    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak", encoding="utf-8")

    sym_key = _make_key("sym-entry")
    (cache_dir / f"{sym_key}.md").symlink_to(secret)
    assert cache.get(sym_key) is None

    dir_key = _make_key("dir-entry")
    (cache_dir / f"{dir_key}.md").mkdir()
    assert cache.get(dir_key) is None


def test_none_directory_disables_cache(tmp_path: Path) -> None:
    cache = CanonicalCache(None, tmp_path)
    key = _make_key("disabled")

    assert cache.enabled is False
    assert cache.directory is None
    assert cache.put(key, "text") is False
    assert cache.get(key) is None
    cache.prune({key})


def test_symlinked_directory_disables_cache(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    real_target = root / "real_target"
    real_target.mkdir(parents=True)

    key = _make_key("existing")
    (real_target / f"{key}.md").write_text("cached", encoding="utf-8")

    # Direct symlinked cache directory (even pointing inside root).
    direct_link = root / ".rag_cache"
    direct_link.symlink_to(real_target)
    cache_direct = CanonicalCache(direct_link, root)
    assert cache_direct.enabled is False
    assert cache_direct.directory is None
    assert cache_direct.get(key) is None
    assert cache_direct.put(_make_key("new"), "payload") is False
    cache_direct.prune(set())
    assert (real_target / f"{key}.md").is_file()

    # Intermediate symlinked parent inside root.
    nested_dir = direct_link / "canonical"
    cache_nested = CanonicalCache(nested_dir, root)
    assert cache_nested.enabled is False
    assert cache_nested.directory is None
    assert cache_nested.put(_make_key("new2"), "payload") is False

    # Directory replaced with a symlink after construction.
    clean_dir = root / "clean_cache"
    clean_dir.mkdir()
    cache_swapped = CanonicalCache(clean_dir, root)
    assert cache_swapped.enabled is True
    clean_dir.rmdir()
    clean_dir.symlink_to(real_target)
    assert cache_swapped.enabled is False
    assert cache_swapped.get(key) is None
    assert cache_swapped.put(_make_key("new3"), "payload") is False
    cache_swapped.prune(set())
    assert (real_target / f"{key}.md").is_file()


def test_directory_resolving_outside_root_disables_cache(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    key = _make_key("outside")
    (outside / f"{key}.md").write_text("outside", encoding="utf-8")

    for bad_dir in (outside, root / ".." / "outside"):
        cache = CanonicalCache(bad_dir, root)
        assert cache.enabled is False
        assert cache.directory is None
        assert cache.get(key) is None
        assert cache.put(_make_key("write"), "nope") is False
        cache.prune(set())
        assert (outside / f"{key}.md").is_file()

    # Symlink inside root resolving outside root.
    escape_link = root / ".rag_cache"
    escape_link.symlink_to(outside)
    cache_escape = CanonicalCache(escape_link / "canonical", root)
    assert cache_escape.enabled is False
    assert cache_escape.put(_make_key("escape"), "nope") is False


def test_readonly_put_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache_dir.mkdir(parents=True)
    cache = CanonicalCache(cache_dir, root)

    existing_key = _make_key("existing")
    assert cache.put(existing_key, "hello") is True

    new_key = _make_key("readonly-new")
    cache_dir.chmod(0o555)
    try:
        # Also simulate PermissionError directly so the check holds even under root UID.
        real_open = os.open

        def _deny_write_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                raise PermissionError("read-only cache directory")
            if dir_fd is not None:
                return real_open(path, flags, mode, dir_fd=dir_fd)
            return real_open(path, flags, mode)

        monkeypatch.setattr(os, "open", _deny_write_open)
        assert cache.put(new_key, "should fail") is False
        assert cache.get(existing_key) == "hello"
        assert not (cache_dir / f"{new_key}.md").exists()
        cache.prune(set())
    finally:
        cache_dir.chmod(0o755)


def test_prune_keeps_foreign_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache_dir.mkdir(parents=True)
    cache = CanonicalCache(cache_dir, root)

    keep_key = _make_key("keep")
    stale_key = _make_key("stale")
    assert cache.put(keep_key, "keep me") is True
    assert cache.put(stale_key, "delete me") is True

    foreign_files = [
        cache_dir / "notes.txt",
        cache_dir / "readme.md",
        cache_dir / f"{'a' * 63}.md",
        cache_dir / f"{'a' * 65}.md",
        cache_dir / f"{'g' * 64}.md",
        cache_dir / f"{keep_key}.tmp",
        cache_dir / f".{keep_key}.deadbeef.tmp",
    ]
    for path in foreign_files:
        path.write_text("foreign", encoding="utf-8")

    hex_subdir = cache_dir / f"{_make_key('subdir')}.md"
    hex_subdir.mkdir()

    cache.prune({keep_key})

    assert (cache_dir / f"{keep_key}.md").is_file()
    assert not (cache_dir / f"{stale_key}.md").exists()
    for path in foreign_files:
        assert path.is_file(), f"foreign file was removed: {path.name}"
    assert hex_subdir.is_dir()


def test_atomic_replace_no_partial_file_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "docs"
    cache_dir = root / ".rag_cache" / "canonical"
    cache = CanonicalCache(cache_dir, root)

    key = _make_key("atomic")
    assert cache.put(key, "original content") is True

    # Before os.replace executes, the target still holds the complete original content
    # (never truncated or partially overwritten). If os.replace fails, no partial or
    # temp file remains.
    observed_during_replace: list[str | None] = []

    def _failing_replace(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        observed_during_replace.append(cache.get(key))
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", _failing_replace)

    assert cache.put(key, "new replacement content") is False
    assert observed_during_replace == ["original content"]
    assert cache.get(key) == "original content"

    new_key = _make_key("atomic-fresh")
    assert cache.put(new_key, "never committed") is False
    assert cache.get(new_key) is None
    assert not (cache_dir / f"{new_key}.md").exists()

    # Temp files must be cleaned up on failure.
    assert [p.name for p in cache_dir.iterdir()] == [f"{key}.md"]


def test_cache_file_0600_and_new_dirs_0700_existing_dirs_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir(mode=0o755)
    rag_cache_dir = root / ".rag_cache"
    cache_dir = rag_cache_dir / "canonical"
    cache = CanonicalCache(cache_dir, root)

    key = _make_key("perms")
    assert cache.put(key, "secret-free canonical text") is True
    assert stat.S_IMODE(rag_cache_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((cache_dir / f"{key}.md").stat().st_mode) == 0o600

    # Existing cache directory permissions must not be modified.
    existing_root = tmp_path / "existing_docs"
    existing_root.mkdir()
    existing_dir = existing_root / ".rag_cache" / "canonical"
    existing_dir.mkdir(parents=True)
    existing_dir.chmod(0o755)
    existing_cache = CanonicalCache(existing_dir, existing_root)
    key2 = _make_key("perms-existing")
    assert existing_cache.put(key2, "payload") is True
    assert stat.S_IMODE(existing_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE((existing_dir / f"{key2}.md").stat().st_mode) == 0o600
