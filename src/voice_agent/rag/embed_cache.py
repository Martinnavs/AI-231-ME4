"""On-disk float16 multi-vector cache for the RAG embedder sidecar."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Duplicated as literals in src/voice_agent/rag/embedder_sidecar.py (which cannot import
# voice_agent because it runs inside the thin embedder venv without rank_bm25).
CACHE_PATH_FORMAT = "{model_slug}/{key}.npy"
CACHE_VECTOR_DTYPE = "float16"


def model_slug(model_name: str) -> str:
    """Return the filesystem-safe directory slug for a Hugging Face model name."""
    return model_name.replace("/", "--")


def embed_key(header: str, text: str) -> str:
    """Return the sha256 hex digest of ``header + '\\n' + text``."""
    payload = f"{header}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _is_safe_key(key: str) -> bool:
    """Reject empty keys or path separators/traversal tokens."""
    if not key or key in (".", ".."):
        return False
    return "/" not in key and "\\" not in key


class EmbedCache:
    """Reader/pruner for per-chunk float16 multi-vector ``.npy`` files written by the sidecar."""

    def __init__(self, cache_dir: Path | str, model_slug_name: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._model_slug = model_slug_name
        self._directory = self._cache_dir / model_slug_name

    @property
    def directory(self) -> Path:
        """Directory holding ``<key>.npy`` vector files for this model."""
        return self._directory

    def path_for(self, key: str) -> Path:
        """Return the expected cache path for ``key``."""
        if not _is_safe_key(key):
            raise ValueError(f"invalid cache key: {key!r}")
        rel = CACHE_PATH_FORMAT.format(model_slug=self._model_slug, key=key)
        return self._cache_dir / rel

    def missing(self, keys: Iterable[str]) -> list[str]:
        """Return keys from ``keys`` whose cache file does not exist on disk."""
        result: list[str] = []
        for key in keys:
            if not _is_safe_key(key):
                result.append(key)
                continue
            if not self.path_for(key).is_file():
                result.append(key)
        return result

    def load(self, key: str) -> NDArray[np.float16] | None:
        """Load a float16 ``(tokens, dim)`` matrix for ``key``, unlinking corrupt files."""
        if not _is_safe_key(key):
            return None
        path = self.path_for(key)
        if not path.is_file():
            return None

        loaded: Any = None
        try:
            loaded = np.load(path, allow_pickle=False)
            if not isinstance(loaded, np.ndarray):
                self._safe_unlink(path)
                return None
            if (
                loaded.dtype != np.dtype(CACHE_VECTOR_DTYPE)
                or loaded.ndim != 2
                or loaded.shape[0] <= 0
                or loaded.shape[1] <= 0
            ):
                self._safe_unlink(path)
                return None
            return loaded
        except Exception:
            if hasattr(loaded, "close"):
                with contextlib.suppress(Exception):
                    loaded.close()
            self._safe_unlink(path)
            return None

    def prune(self, keep: AbstractSet[str]) -> None:
        """Unlink ``*.npy`` files in ``directory`` whose stem is not in ``keep``."""
        try:
            if not self._directory.is_dir():
                return
            entries = list(self._directory.iterdir())
        except OSError:
            return

        for entry in entries:
            try:
                if entry.suffix != ".npy" or not entry.is_file():
                    continue
                if entry.stem not in keep:
                    self._safe_unlink(entry)
            except OSError:
                continue

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
