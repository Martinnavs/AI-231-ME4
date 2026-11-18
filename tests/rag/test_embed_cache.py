"""Offline unit tests for voice_agent.rag.embed_cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from voice_agent.rag.embed_cache import (
    CACHE_PATH_FORMAT,
    CACHE_VECTOR_DTYPE,
    EmbedCache,
    embed_key,
    model_slug,
)


def test_embed_key_varies_with_header_and_matches_sha256() -> None:
    text = "Identical passage text."
    k1 = embed_key("guide.md > Intro", text)
    k2 = embed_key("guide.md > Appendix", text)
    assert k1 != k2
    assert len(k1) == 64
    assert k1 == hashlib.sha256(b"guide.md > Intro\nIdentical passage text.").hexdigest()
    assert model_slug("answerdotai/answerai-colbert-small-v1") == (
        "answerdotai--answerai-colbert-small-v1"
    )
    assert CACHE_PATH_FORMAT == "{model_slug}/{key}.npy"
    assert CACHE_VECTOR_DTYPE == "float16"


def test_missing_load_and_prune(tmp_path: Path) -> None:
    slug = model_slug("org/model")
    cache = EmbedCache(tmp_path, slug)
    assert cache.directory == tmp_path / slug
    cache.directory.mkdir(parents=True)

    arr_a = np.ones((3, 16), dtype=np.float16)
    arr_b = np.full((2, 16), 2.5, dtype=np.float16)
    np.save(cache.directory / "key_a.npy", arr_a, allow_pickle=False)
    np.save(cache.directory / "key_b.npy", arr_b, allow_pickle=False)
    (cache.directory / "notes.txt").write_text("keep non-npy", encoding="utf-8")

    assert cache.missing(["key_a", "key_b", "key_c"]) == ["key_c"]

    loaded = cache.load("key_a")
    assert loaded is not None
    assert loaded.dtype == np.float16
    assert loaded.shape == (3, 16)
    assert np.allclose(loaded, arr_a)

    cache.prune({"key_a"})
    assert (cache.directory / "key_a.npy").is_file()
    assert not (cache.directory / "key_b.npy").exists()
    assert (cache.directory / "notes.txt").is_file()
    assert cache.missing(["key_a", "key_b"]) == ["key_b"]


def test_corrupt_and_wrong_shape_files_are_unlinked_and_reported_missing(tmp_path: Path) -> None:
    cache = EmbedCache(tmp_path, "org--model")
    cache.directory.mkdir(parents=True)

    corrupt_path = cache.directory / "corrupt.npy"
    corrupt_path.write_bytes(b"not a valid npy payload")

    one_d_path = cache.directory / "wrong_1d.npy"
    np.save(one_d_path, np.ones((16,), dtype=np.float16), allow_pickle=False)

    empty_2d_path = cache.directory / "wrong_empty.npy"
    np.save(empty_2d_path, np.zeros((0, 16), dtype=np.float16), allow_pickle=False)

    wrong_dtype_path = cache.directory / "wrong_dtype.npy"
    np.save(wrong_dtype_path, np.ones((2, 16), dtype=np.float32), allow_pickle=False)

    for key in ("corrupt", "wrong_1d", "wrong_empty", "wrong_dtype"):
        assert cache.load(key) is None

    for path in (corrupt_path, one_d_path, empty_2d_path, wrong_dtype_path):
        assert not path.exists()

    assert cache.missing(["corrupt", "wrong_1d", "wrong_empty", "wrong_dtype"]) == [
        "corrupt",
        "wrong_1d",
        "wrong_empty",
        "wrong_dtype",
    ]


def test_readonly_directory_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = EmbedCache(tmp_path, "org--model")
    cache.directory.mkdir(parents=True)
    bad = cache.directory / "bad.npy"
    bad.write_bytes(b"broken")
    stale = cache.directory / "stale.npy"
    np.save(stale, np.ones((2, 16), dtype=np.float16), allow_pickle=False)

    cache.directory.chmod(0o555)
    try:
        # Also simulate PermissionError directly so the check holds even under root UID.
        def _deny_unlink(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("read-only directory")

        monkeypatch.setattr(Path, "unlink", _deny_unlink)
        assert cache.load("bad") is None
        cache.prune({"other"})
    finally:
        cache.directory.chmod(0o755)
