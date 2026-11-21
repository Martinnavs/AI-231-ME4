"""On-disk cache for canonical Markdown documents."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import secrets
import stat
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from pathlib import Path

MAX_CANONICAL_BYTES = 64_000_000

_HEX64_RE = re.compile(r"[0-9a-fA-F]{64}", re.ASCII)
_HEX64_MD_RE = re.compile(r"([0-9a-fA-F]{64})\.md", re.ASCII)

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC


def cache_key(
    source_sha256: str,
    kind: str,
    converter_id: str,
    converter_version: str,
    postproc_version: int | str,
) -> str:
    """Return the sha256 hex digest identifying a canonical conversion result."""
    payload = (
        f"{source_sha256}|{kind}|{converter_id}|{converter_version}|{postproc_version}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_valid_key(key: str) -> bool:
    return bool(_HEX64_RE.fullmatch(key))


def _validate_directory(dir_path: Path, root_path: Path) -> tuple[Path, tuple[str, ...]] | None:
    """Validate that ``dir_path`` is non-symlinked and resolves inside ``root_path``."""
    try:
        if dir_path.is_symlink():
            return None
        if dir_path.exists() and not dir_path.is_dir():
            return None
        root_resolved = root_path.resolve()
        dir_resolved = dir_path.resolve()
    except (OSError, RuntimeError):
        return None

    if not dir_resolved.is_relative_to(root_resolved):
        return None

    dir_abs = Path(os.path.abspath(dir_path))
    root_abs = Path(os.path.abspath(root_path))

    cur = dir_abs
    while True:
        if cur == root_abs:
            break
        try:
            if cur.is_symlink():
                return None
            if cur.resolve() == root_resolved:
                break
        except (OSError, RuntimeError):
            return None
        if cur == cur.parent:
            break
        cur = cur.parent

    if dir_abs.is_relative_to(root_abs):
        return root_abs, dir_abs.relative_to(root_abs).parts
    return root_resolved, dir_resolved.relative_to(root_resolved).parts


validate_cache_dir = _validate_directory


class CanonicalCache:
    """Symlink-safe, atomic on-disk cache for canonical Markdown strings."""

    def __init__(self, directory: Path | str | None, root: Path | str) -> None:
        self._root = Path(root)
        self._directory: Path | None = None
        self._base_path: Path | None = None
        self._rel_parts: tuple[str, ...] = ()

        if directory is None:
            return

        dir_path = Path(directory)
        validated = _validate_directory(dir_path, self._root)
        if validated is None:
            return

        self._directory = dir_path
        self._base_path, self._rel_parts = validated

    @property
    def directory(self) -> Path | None:
        """Cache directory, or ``None`` when caching is disabled."""
        return self._directory

    @property
    def root(self) -> Path:
        """Root directory within which the cache directory must resolve."""
        return self._root

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled."""
        return self._directory is not None and not self._directory.is_symlink()

    def _open_cache_dir(self, *, create: bool) -> int | None:
        if self._directory is None or self._base_path is None:
            return None
        if self._directory.is_symlink():
            return None

        base_flags = (
            _DIR_FLAGS if not self._rel_parts else (os.O_RDONLY | os.O_DIRECTORY | _O_CLOEXEC)
        )
        try:
            fd = os.open(str(self._base_path), base_flags)
        except FileNotFoundError:
            if not create:
                return None
            try:
                self._base_path.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(str(self._base_path), base_flags)
            except OSError:
                return None
        except OSError:
            return None

        try:
            for part in self._rel_parts:
                try:
                    nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        os.close(fd)
                        return None
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(part, 0o700, dir_fd=fd)
                    nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = nxt
        except OSError:
            os.close(fd)
            return None

        return fd

    def get(self, key: str) -> str | None:
        """Return cached canonical Markdown for ``key``, or ``None`` on miss/invalid entry."""
        if not _is_valid_key(key):
            return None

        dfd = self._open_cache_dir(create=False)
        if dfd is None:
            return None

        try:
            fd = os.open(
                f"{key}.md",
                os.O_RDONLY | os.O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
                dir_fd=dfd,
            )
        except OSError:
            return None
        finally:
            os.close(dfd)

        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_CANONICAL_BYTES:
                return None
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(MAX_CANONICAL_BYTES + 1)
            if len(raw) > MAX_CANONICAL_BYTES:
                return None
            return raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        finally:
            os.close(fd)

    def put(self, key: str, text: str) -> bool:
        """Atomically write ``text`` for ``key``. Swallows ``OSError`` and returns success."""
        if not _is_valid_key(key):
            return False

        encoded = text.encode("utf-8")
        if len(encoded) > MAX_CANONICAL_BYTES:
            return False

        dfd = self._open_cache_dir(create=True)
        if dfd is None:
            return False

        tmp_name = f".{key}.{secrets.token_hex(8)}.tmp"
        try:
            try:
                fd = os.open(
                    tmp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC,
                    0o600,
                    dir_fd=dfd,
                )
            except OSError:
                return False

            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name, dir_fd=dfd)
                raise

            try:
                with handle:
                    handle.write(encoded)
                    handle.flush()
                os.replace(tmp_name, f"{key}.md", src_dir_fd=dfd, dst_dir_fd=dfd)
                return True
            except OSError:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name, dir_fd=dfd)
                return False
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name, dir_fd=dfd)
                raise
        finally:
            os.close(dfd)

    def prune(self, keep: Iterable[str]) -> None:
        """Delete unkept ``<64 hex>.md`` regular files in the cache directory."""
        keep_set = keep if isinstance(keep, AbstractSet) else set(keep)
        dfd = self._open_cache_dir(create=False)
        if dfd is None:
            return

        try:
            with os.scandir(dfd) as entries:
                for entry in entries:
                    match = _HEX64_MD_RE.fullmatch(entry.name)
                    if match is None:
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if match.group(1) not in keep_set:
                            os.unlink(entry.name, dir_fd=dfd)
                    except OSError:
                        continue
        except OSError:
            return
        finally:
            os.close(dfd)
