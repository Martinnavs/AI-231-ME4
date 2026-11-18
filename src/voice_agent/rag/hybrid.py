"""Hybrid retriever combining BM25 with late-interaction MaxSim via an embedder sidecar."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from voice_agent.rag.docindex import Bm25Retriever, Chunk, Retriever
from voice_agent.rag.embed_cache import EmbedCache, embed_key, model_slug
from voice_agent.rag.embed_client import SidecarClient, SidecarError
from voice_agent.rag.embedder_sidecar import DEFAULT_MODEL, DEFAULT_MODEL_DIR, DEFAULT_THREADS

RetrieverMode = Literal["bm25", "hybrid", "auto"]
ALLOWED_MODES: tuple[RetrieverMode, ...] = ("bm25", "hybrid", "auto")
DEFAULT_EMBED_PYTHON = "~/.venvs/rag-embed/bin/python"
DEFAULT_SIDECAR_SCRIPT = Path(__file__).resolve().with_name("embedder_sidecar.py")
EMBED_BATCH_SIZE = 16
MAX_CONSECUTIVE_QUERY_FAILURES = 2


def resolve_mode(
    flag: str | None = None,
    env: Mapping[str, str] | None = None,
) -> RetrieverMode:
    """Resolve retrieval mode: ``flag`` beats ``VOICE_AGENT_RAG_RETRIEVAL`` beats ``auto``."""
    env_map = os.environ if env is None else env
    raw: str | None
    if flag is not None:
        raw = flag.strip()
    else:
        env_val = env_map.get("VOICE_AGENT_RAG_RETRIEVAL")
        raw = env_val.strip() if env_val is not None and env_val.strip() else None

    if raw is None:
        return "auto"
    normalized = raw.lower()
    if normalized not in ALLOWED_MODES:
        allowed = ", ".join(ALLOWED_MODES)
        raise ValueError(f"unknown retriever mode {raw!r}; expected one of: {allowed}")
    return normalized


def _clean_reason(reason: str) -> str:
    single = " ".join(reason.splitlines()).strip()
    return single or "unknown error"


class HybridRetriever:
    """Retriever serving BM25 immediately while warming late-interaction MaxSim vectors."""

    def __init__(
        self,
        client: SidecarClient | None,
        cache: EmbedCache | None,
        mode: RetrieverMode = "auto",
        rrf_k: int = 60,
        bm25: Retriever | None = None,
        background: bool = True,
        *,
        model_name: str | None = None,
    ) -> None:
        if mode not in ALLOWED_MODES:
            allowed = ", ".join(ALLOWED_MODES)
            raise ValueError(f"unknown retriever mode {mode!r}; expected one of: {allowed}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")

        self._client = client
        self._cache = cache
        self._mode: RetrieverMode = mode
        self._rrf_k = int(rrf_k)
        self._bm25: Retriever = bm25 if bm25 is not None else Bm25Retriever()
        self._background = bool(background)
        self._model_name = (
            model_name
            or getattr(client, "model", None)
            or getattr(client, "_model", None)
            or DEFAULT_MODEL
        )

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

        self._generation = 0
        self._chunks: list[Chunk] = []
        self._keys: list[str] = []
        self._ready = False
        self._checked_ok = False
        self._broken_reason: str | None = None
        self._logged_unavailable = False
        self._consecutive_query_failures = 0
        self._done = 0
        self._total = 0
        self._closed = False

    @property
    def ready(self) -> bool:
        """True once dense vectors for the current index are embedded and serving."""
        with self._lock:
            return self._ready and self._broken_reason is None

    def status(self) -> str:
        """Return the current retriever state string."""
        with self._lock:
            if self._mode == "bm25":
                return "bm25"
            if self._broken_reason is not None:
                if self._mode == "hybrid":
                    return f"bm25 (hybrid unavailable: {self._broken_reason})"
                return "bm25"
            if self._ready:
                return f"hybrid:{self._model_name}"
            return f"bm25 (hybrid warming: {self._done}/{self._total})"

    def _mark_unavailable_locked(self, reason: str) -> None:
        cleaned = _clean_reason(reason)
        if self._broken_reason is None:
            self._broken_reason = cleaned
        self._ready = False
        if not self._logged_unavailable:
            self._logged_unavailable = True
            print(f"rag: hybrid unavailable ({cleaned}); serving BM25", file=sys.stderr, flush=True)
        self._cond.notify_all()

    def _mark_unavailable(self, reason: str) -> None:
        with self._lock:
            self._mark_unavailable_locked(reason)
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()

    def _is_stale(self, gen: int) -> bool:
        with self._lock:
            return self._closed or self._generation != gen or self._broken_reason is not None

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Build BM25 immediately and warm dense passage embeddings in the background."""
        chunk_list = list(chunks)
        keys = [embed_key(chunk.header, chunk.text) for chunk in chunk_list]
        self._bm25.build(chunk_list)

        with self._lock:
            self._generation += 1
            gen = self._generation
            self._chunks = chunk_list
            self._keys = keys
            self._ready = False
            self._done = 0
            self._total = len(chunk_list)
            self._cond.notify_all()

            if self._closed or self._mode == "bm25" or self._broken_reason is not None:
                return
            if self._client is None:
                self._mark_unavailable_locked("no embedder client")
                return
            if self._cache is None:
                self._mark_unavailable_locked("no cache directory")
                return

        if self._background:
            thread = threading.Thread(
                target=self._build_worker,
                args=(gen, chunk_list, keys),
                name=f"rag-hybrid-build-{gen}",
                daemon=True,
            )
            with self._lock:
                if self._generation != gen or self._closed:
                    return
                self._worker_thread = thread
            thread.start()
        else:
            self._build_worker(gen, chunk_list, keys)

    def _build_worker(self, gen: int, chunk_list: list[Chunk], keys: list[str]) -> None:
        if self._is_stale(gen):
            return

        with self._worker_lock:
            if self._is_stale(gen):
                return

            client = self._client
            cache = self._cache
            if client is None or cache is None:
                return

            try:
                if not self._checked_ok:
                    check_err = client.check()
                    if check_err is not None:
                        if not self._is_stale(gen):
                            self._mark_unavailable(check_err)
                        return
                    self._checked_ok = True

                if self._is_stale(gen):
                    return

                if not chunk_list:
                    cache.prune(set())
                    with self._lock:
                        if (
                            not self._closed
                            and self._generation == gen
                            and self._broken_reason is None
                        ):
                            self._done = 0
                            self._total = 0
                            self._ready = True
                            self._cond.notify_all()
                    return

                client.start()
                if self._is_stale(gen):
                    return

                key_to_text: dict[str, str] = {}
                for chunk, key in zip(chunk_list, keys, strict=True):
                    if key not in key_to_text:
                        key_to_text[key] = f"{chunk.header}\n{chunk.text}"

                missing_keys = list(dict.fromkeys(cache.missing(keys)))
                total_unique = len(key_to_text)
                already_done = total_unique - len(missing_keys)
                with self._lock:
                    if self._closed or self._generation != gen or self._broken_reason is not None:
                        return
                    self._total = total_unique
                    self._done = already_done

                for start in range(0, len(missing_keys), EMBED_BATCH_SIZE):
                    if self._is_stale(gen):
                        return
                    batch_keys = missing_keys[start : start + EMBED_BATCH_SIZE]
                    batch_texts = [key_to_text[k] for k in batch_keys]
                    done_keys = client.embed_passages(batch_keys, batch_texts)
                    if len(done_keys) != len(batch_keys) or set(done_keys) != set(batch_keys):
                        raise SidecarError(
                            f"embed_passages returned mismatched keys: "
                            f"expected {len(batch_keys)}, got {len(done_keys)}"
                        )
                    with self._lock:
                        if self._closed or self._generation != gen:
                            return
                        self._done += len(batch_keys)

                if self._is_stale(gen):
                    return

                cache.prune(set(keys))
                with self._lock:
                    if not self._closed and self._generation == gen and self._broken_reason is None:
                        self._consecutive_query_failures = 0
                        self._ready = True
                        self._cond.notify_all()
            except (SidecarError, OSError, ValueError) as exc:
                if not self._is_stale(gen):
                    self._mark_unavailable(str(exc))
            except Exception as exc:
                if not self._is_stale(gen):
                    self._mark_unavailable(f"unexpected embedder error: {exc}")

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` seconds for the current build's dense index to be ready."""
        with self._cond:
            deadline = None if timeout is None else (time.monotonic() + max(0.0, float(timeout)))
            while True:
                if self._ready and self._broken_reason is None:
                    return True
                if self._closed or self._mode == "bm25" or self._broken_reason is not None:
                    return False
                if self._worker_thread is None or not self._worker_thread.is_alive():
                    return self._ready and self._broken_reason is None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return self._ready and self._broken_reason is None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Rank chunks by BM25, or BM25 + MaxSim via RRF when dense vectors are ready."""
        if k <= 0 or not query.strip():
            return []

        with self._lock:
            chunks_len = len(self._chunks)
            if chunks_len == 0:
                return []
            gen = self._generation
            keys = list(self._keys)
            use_dense = (
                not self._closed
                and self._mode != "bm25"
                and self._ready
                and self._broken_reason is None
                and self._client is not None
            )

        bm25_ranking = self._bm25.rank(query, chunks_len)
        if not use_dense:
            return bm25_ranking[:k]

        assert self._client is not None
        try:
            dense_ranking = self._client.score_query(query, keys)
            self._validate_dense_ranking(dense_ranking, chunks_len)
        except Exception as exc:
            self._record_query_failure(gen, str(exc))
            return bm25_ranking[:k]

        with self._lock:
            if self._closed or self._generation != gen or self._broken_reason is not None:
                return self._bm25.rank(query, k)
            self._consecutive_query_failures = 0

        return self._fuse_rrf(bm25_ranking, dense_ranking)[:k]

    @staticmethod
    def _validate_dense_ranking(
        dense_ranking: Sequence[tuple[int, float]],
        chunks_len: int,
    ) -> None:
        if chunks_len > 0 and not dense_ranking:
            raise SidecarError("score_query returned empty ranking for non-empty index")
        seen: set[int] = set()
        for idx, score in dense_ranking:
            if type(idx) is not int or idx < 0 or idx >= chunks_len or idx in seen:
                raise SidecarError(f"score_query returned invalid chunk index: {idx!r}")
            if math.isnan(score):
                raise SidecarError("score_query returned NaN score")
            seen.add(idx)

    def _record_query_failure(self, gen: int, reason: str) -> None:
        should_close = False
        with self._lock:
            if self._closed or self._generation != gen or self._broken_reason is not None:
                return
            self._consecutive_query_failures += 1
            if self._consecutive_query_failures >= MAX_CONSECUTIVE_QUERY_FAILURES:
                self._mark_unavailable_locked(reason)
                should_close = True
        if should_close and self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()

    def _fuse_rrf(
        self,
        bm25_ranking: Sequence[tuple[int, float]],
        dense_ranking: Sequence[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for rank_pos, (idx, _score) in enumerate(bm25_ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self._rrf_k + rank_pos)
        for rank_pos, (idx, score) in enumerate(dense_ranking, start=1):
            if math.isfinite(score):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self._rrf_k + rank_pos)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return ordered

    def close(self) -> None:
        """Stop any warming worker and close the embedder sidecar process."""
        with self._lock:
            self._closed = True
            self._generation += 1
            self._ready = False
            worker = self._worker_thread
            self._cond.notify_all()

        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()

        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)


def make_retriever_factory(
    mode: RetrieverMode = "auto",
    cache_dir: Path | str | None = None,
    embed_python: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    model_dir: Path | str | None = None,
    script: Path | str | None = None,
    *,
    background: bool = True,
    rrf_k: int = 60,
    threads: int = DEFAULT_THREADS,
) -> Callable[[], Retriever]:
    """Build a ``Retriever`` factory for ``DocIndex``; hybrid/auto reuse one instance."""
    resolved_mode = resolve_mode(mode, {})
    if resolved_mode == "bm25":
        return Bm25Retriever

    raw_python = (
        str(embed_python).strip()
        if embed_python is not None and str(embed_python).strip()
        else (os.environ.get("VOICE_AGENT_EMBED_PYTHON") or "").strip() or DEFAULT_EMBED_PYTHON
    )
    eff_python = str(Path(raw_python).expanduser())
    eff_script = Path(script).expanduser() if script is not None else DEFAULT_SIDECAR_SCRIPT
    eff_model_dir = (
        Path(model_dir).expanduser()
        if model_dir is not None
        else Path(DEFAULT_MODEL_DIR).expanduser()
    )
    eff_cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None

    cache = EmbedCache(eff_cache_dir, model_slug(model)) if eff_cache_dir is not None else None
    client = SidecarClient(
        python=eff_python,
        script=eff_script,
        cache_dir=eff_cache_dir if eff_cache_dir is not None else Path("."),
        model=model,
        model_dir=eff_model_dir,
        threads=threads,
    )
    retriever = HybridRetriever(
        client=client,
        cache=cache,
        mode=resolved_mode,
        rrf_k=rrf_k,
        background=background,
        model_name=model,
    )

    def _factory() -> Retriever:
        return retriever

    return _factory
