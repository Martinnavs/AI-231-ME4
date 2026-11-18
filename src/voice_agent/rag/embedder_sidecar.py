"""Stdio JSON-lines late-interaction embedder sidecar script.

This module MUST be executed as a standalone script by file path (never via
``python -m voice_agent.rag.embedder_sidecar``) because ``voice_agent.rag.__init__``
imports ``rank_bm25``, which is absent from the thin ``~/.venvs/rag-embed`` venv.
It imports only the Python standard library, ``numpy``, and (lazily inside the
``fastembed`` backend) ``fastembed``.

Backends:
- ``fastembed`` (default): ``LateInteractionTextEmbedding`` with
  ``local_files_only=True`` during serving and ``--check``, and ``HF_HUB_OFFLINE=1``
  forced in serve mode so serving can never reach the network.
- ``hash`` (TEST-ONLY): deterministic numpy multi-vectors of dimension 16 built from
  hashed tokens, allowing offline tests to exercise the full protocol, on-disk
  float16 cache layout, and MaxSim scoring with no model or network. Setting
  ``VOICE_AGENT_SIDECAR_TEST_NOISE=1`` (TEST-ONLY) makes the ``hash`` backend print
  to stdout/fd 1 to verify protocol isolation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

# Duplicated as literals from src/voice_agent/rag/embed_cache.py because this script
# cannot import voice_agent in the thin embedder venv.
CACHE_PATH_FORMAT = "{model_slug}/{key}.npy"
CACHE_VECTOR_DTYPE = "float16"

DEFAULT_MODEL = "answerdotai/answerai-colbert-small-v1"
DEFAULT_MODEL_DIR = "~/.cache/voice_agent/fastembed"
DEFAULT_CACHE_DIR = ".rag_cache"
DEFAULT_THREADS = 4
MAX_LINE_BYTES = 16 * 1024 * 1024
_HASH_DIM = 16
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _EmbedBackend(Protocol):
    @property
    def dim(self) -> int: ...

    def passage_embed(self, texts: Sequence[str]) -> Iterable[NDArray[np.float32]]: ...

    def query_embed(self, query: str) -> NDArray[np.float32]: ...


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def _is_safe_key(key: str) -> bool:
    if not key or key in (".", ".."):
        return False
    return "/" not in key and "\\" not in key


def _target_path(cache_dir: Path, slug: str, key: str) -> Path:
    if not _is_safe_key(key):
        raise ValueError(f"invalid cache key: {key!r}")
    return cache_dir / CACHE_PATH_FORMAT.format(model_slug=slug, key=key)


class _HashBackend:
    """Test-only deterministic hashed-token multi-vector backend (dim=16)."""

    def __init__(self) -> None:
        self._dim = _HASH_DIM
        self._noise = os.environ.get("VOICE_AGENT_SIDECAR_TEST_NOISE") == "1"
        if self._noise:
            print("test-only hash backend init stdout noise")
            os.write(1, b"test-only hash backend init fd1 noise\n")

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_tokens(self, text: str) -> NDArray[np.float32]:
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return np.zeros((1, self._dim), dtype=np.float32)
        rows: list[NDArray[np.float32]] = []
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            raw = np.frombuffer(digest[: self._dim], dtype=np.uint8).astype(np.float32) - 127.5
            norm = float(np.linalg.norm(raw))
            if norm > 0.0:
                raw = raw / norm
            rows.append(raw)
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    def passage_embed(self, texts: Sequence[str]) -> Iterable[NDArray[np.float32]]:
        if self._noise:
            print("test-only hash backend passage_embed stdout noise")
        for text in texts:
            yield self._encode_tokens(text)

    def query_embed(self, query: str) -> NDArray[np.float32]:
        if self._noise:
            print("test-only hash backend query_embed stdout noise")
        return self._encode_tokens(query)


class _FastembedBackend:
    """Lazy fastembed ``LateInteractionTextEmbedding`` wrapper."""

    def __init__(
        self,
        model_name: str,
        model_dir: Path,
        threads: int,
        *,
        local_files_only: bool,
    ) -> None:
        from fastembed import (  # type: ignore[import-not-found,import-untyped]
            LateInteractionTextEmbedding,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = LateInteractionTextEmbedding(
            model_name=model_name,
            cache_dir=str(model_dir),
            threads=threads,
            local_files_only=local_files_only,
        )
        probe = next(iter(self._model.query_embed(["dim"])))
        probe_arr = np.asarray(probe, dtype=np.float32)
        if probe_arr.ndim != 2 or probe_arr.shape[1] <= 0:
            raise RuntimeError(f"unexpected embedding shape from {model_name}: {probe_arr.shape}")
        self._dim = int(probe_arr.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def passage_embed(self, texts: Sequence[str]) -> Iterable[NDArray[np.float32]]:
        for vec in self._model.passage_embed(list(texts)):
            yield np.asarray(vec, dtype=np.float32)

    def query_embed(self, query: str) -> NDArray[np.float32]:
        vec = next(iter(self._model.query_embed([query])))
        return np.asarray(vec, dtype=np.float32)


def _build_backend(
    backend_name: str,
    model_name: str,
    model_dir: Path,
    threads: int,
    *,
    local_files_only: bool,
) -> _EmbedBackend:
    if backend_name == "hash":
        return _HashBackend()
    if backend_name == "fastembed":
        return _FastembedBackend(
            model_name=model_name,
            model_dir=model_dir,
            threads=threads,
            local_files_only=local_files_only,
        )
    raise ValueError(f"unknown backend: {backend_name!r}")


def _write_atomic_npy(target: Path, arr: NDArray[np.float16]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.stem}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, arr, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def _load_cached_float32(path: Path, expected_dim: int) -> NDArray[np.float32] | None:
    if not path.is_file():
        return None
    loaded: Any = None
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            return None
        if (
            loaded.dtype != np.dtype(CACHE_VECTOR_DTYPE)
            or loaded.ndim != 2
            or loaded.shape[0] <= 0
            or loaded.shape[1] != expected_dim
        ):
            return None
        return loaded.astype(np.float32, copy=False)
    except Exception:
        if hasattr(loaded, "close"):
            with contextlib.suppress(Exception):
                loaded.close()
        return None


def _dir_size_bytes(directory: Path) -> int:
    total = 0
    if not directory.exists():
        return 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _error_response(req_id: int, message: str) -> dict[str, Any]:
    return {"id": req_id, "ok": False, "error": message}


def _extract_req_id(payload: dict[str, Any]) -> int:
    raw_id = payload.get("id")
    if type(raw_id) is int:
        return raw_id
    return -1


def _redirect_stdout_to_stderr() -> Any:
    """Dup fd 1 for protocol JSON lines and point sys.stdout and fd 1 to stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    proto_out = os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    return proto_out


def _read_bounded_line(stream: Any, max_bytes: int) -> tuple[bytes, bool]:
    """Read one line up to ``max_bytes``. Returns ``(line_bytes, too_large)``."""
    raw: bytes = stream.readline(max_bytes + 1)
    if not raw:
        return b"", False
    if len(raw) > max_bytes and not raw.endswith(b"\n"):
        # Drain the rest of the oversized line so the stream stays line-aligned.
        while True:
            chunk: bytes = stream.readline(65536)
            if not chunk or chunk.endswith(b"\n"):
                break
        return raw, True
    if len(raw) > max_bytes:
        return raw, True
    return raw, False


def _serve(
    cache_dir: Path,
    model_name: str,
    model_dir: Path,
    threads: int,
    backend_name: str,
) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    proto_out = _redirect_stdout_to_stderr()

    slug = _model_slug(model_name)
    backend: _EmbedBackend | None = None
    vector_cache: dict[str, NDArray[np.float32]] = {}
    download_cmd = (
        f"~/.venvs/rag-embed/bin/python {Path(__file__).resolve()} --download"
        f" --model {model_name} --model-dir {model_dir}"
    )

    def ensure_backend() -> tuple[_EmbedBackend | None, str | None]:
        nonlocal backend
        if backend is not None:
            return backend, None
        try:
            backend = _build_backend(
                backend_name=backend_name,
                model_name=model_name,
                model_dir=model_dir,
                threads=threads,
                local_files_only=True,
            )
            return backend, None
        except Exception as exc:
            return None, f"model_missing: {exc}; run {download_cmd}"

    def write_response(resp: dict[str, Any]) -> None:
        encoded = json.dumps(resp) + "\n"
        proto_out.write(encoded)
        proto_out.flush()

    stdin_buf = sys.stdin.buffer
    while True:
        raw, too_large = _read_bounded_line(stdin_buf, MAX_LINE_BYTES)
        if not raw:
            break
        if too_large:
            write_response(_error_response(-1, "line_too_large"))
            continue

        stripped = raw.strip()
        if not stripped:
            write_response(_error_response(-1, "malformed_json: empty line"))
            continue

        try:
            req = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            write_response(_error_response(-1, f"malformed_json: {exc}"))
            continue

        if not isinstance(req, dict):
            write_response(_error_response(-1, "malformed_request: expected JSON object"))
            continue

        req_id = _extract_req_id(req)
        op = req.get("op")
        if not isinstance(op, str):
            write_response(_error_response(req_id, "malformed_request: missing string op"))
            continue

        if op == "ping":
            loaded, err = ensure_backend()
            if loaded is None:
                write_response(_error_response(req_id, err or "model_missing"))
                continue
            write_response(
                {
                    "id": req_id,
                    "ok": True,
                    "model": model_name,
                    "dim": loaded.dim,
                    "backend": backend_name,
                    "pid": os.getpid(),
                }
            )
            continue

        if op == "embed_passages":
            keys = req.get("keys")
            texts = req.get("texts")
            if (
                not isinstance(keys, list)
                or not isinstance(texts, list)
                or len(keys) != len(texts)
                or not all(isinstance(k, str) and _is_safe_key(k) for k in keys)
                or not all(isinstance(t, str) for t in texts)
            ):
                write_response(_error_response(req_id, "invalid_arguments: keys/texts mismatch"))
                continue

            loaded, err = ensure_backend()
            if loaded is None:
                write_response(_error_response(req_id, err or "model_missing"))
                continue

            try:
                done: list[str] = []
                if keys:
                    vectors = loaded.passage_embed(texts)
                    for key, vec in zip(keys, vectors, strict=True):
                        arr_f16 = np.asarray(vec, dtype=np.dtype(CACHE_VECTOR_DTYPE))
                        if arr_f16.ndim == 1:
                            arr_f16 = arr_f16.reshape(1, -1)
                        if arr_f16.ndim != 2 or arr_f16.shape[1] != loaded.dim:
                            raise ValueError(
                                f"unexpected passage vector shape {arr_f16.shape} for {key!r}"
                            )
                        _write_atomic_npy(_target_path(cache_dir, slug, key), arr_f16)
                        vector_cache.pop(key, None)
                        done.append(key)
                write_response({"id": req_id, "ok": True, "done": done})
            except Exception as exc:
                write_response(_error_response(req_id, f"embed_failed: {exc}"))
            continue

        if op == "score_query":
            query = req.get("query")
            keys = req.get("keys")
            if (
                not isinstance(query, str)
                or not isinstance(keys, list)
                or not all(isinstance(k, str) for k in keys)
            ):
                write_response(_error_response(req_id, "invalid_arguments: query/keys"))
                continue

            loaded, err = ensure_backend()
            if loaded is None:
                write_response(_error_response(req_id, err or "model_missing"))
                continue

            try:
                q_vec = loaded.query_embed(query)
                if q_vec.ndim == 1:
                    q_vec = q_vec.reshape(1, -1)
                q_f32 = np.asarray(q_vec, dtype=np.float32)

                scored: list[tuple[int, float]] = []
                missing: list[str] = []
                for pos, key in enumerate(keys):
                    if not _is_safe_key(key):
                        missing.append(key)
                        scored.append((pos, float("-inf")))
                        continue
                    p_f32 = vector_cache.get(key)
                    if p_f32 is None:
                        p_f32 = _load_cached_float32(_target_path(cache_dir, slug, key), loaded.dim)
                        if p_f32 is not None:
                            vector_cache[key] = p_f32
                    if p_f32 is None or q_f32.shape[0] == 0 or p_f32.shape[1] != q_f32.shape[1]:
                        missing.append(key)
                        scored.append((pos, float("-inf")))
                        continue
                    sim = q_f32 @ p_f32.T
                    score = float(sim.max(axis=1).sum())
                    scored.append((pos, score))

                scored.sort(key=lambda item: (-item[1], item[0]))
                write_response(
                    {
                        "id": req_id,
                        "ok": True,
                        "ranking": [[pos, score] for pos, score in scored],
                        "missing": missing,
                    }
                )
            except Exception as exc:
                write_response(_error_response(req_id, f"score_failed: {exc}"))
            continue

        write_response(_error_response(req_id, f"unknown_op: {op}"))

    return 0


def _run_check(model_name: str, model_dir: Path, threads: int, backend_name: str) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        loaded = _build_backend(
            backend_name=backend_name,
            model_name=model_name,
            model_dir=model_dir,
            threads=threads,
            local_files_only=True,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps({"ok": True, "model": model_name, "dim": loaded.dim}))
    return 0


def _run_download(model_name: str, model_dir: Path, threads: int, backend_name: str) -> int:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        loaded = _build_backend(
            backend_name=backend_name,
            model_name=model_name,
            model_dir=model_dir,
            threads=threads,
            local_files_only=False,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "model": model_name,
                "model_dir": str(model_dir),
                "size_bytes": _dir_size_bytes(model_dir),
                "dim": loaded.dim,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the embedder sidecar."""
    parser = argparse.ArgumentParser(description="Late-interaction embedder sidecar")
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--backend", choices=("fastembed", "hash"), default="fastembed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify local model files and exit")
    mode.add_argument(
        "--download",
        action="store_true",
        help="Download model into --model-dir (only network path) and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for CLI check, download, or stdio serve loop."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    model_dir = Path(args.model_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()

    if args.check:
        return _run_check(args.model, model_dir, args.threads, args.backend)
    if args.download:
        return _run_download(args.model, model_dir, args.threads, args.backend)
    return _serve(cache_dir, args.model, model_dir, args.threads, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
