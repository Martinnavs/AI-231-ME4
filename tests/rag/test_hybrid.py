"""Offline unit and integration tests for voice_agent.rag.hybrid."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import metrics
from voice_agent.rag.docindex import Bm25Retriever, Chunk, DocIndex
from voice_agent.rag.embed_cache import EmbedCache, embed_key, model_slug
from voice_agent.rag.embed_client import SidecarClient
from voice_agent.rag.embedder_sidecar import DEFAULT_MODEL
from voice_agent.rag.hybrid import HybridRetriever, make_retriever_factory, resolve_mode

REAL_SIDECAR = (
    Path(__file__).resolve().parents[2] / "src" / "voice_agent" / "rag" / "embedder_sidecar.py"
)
FAKE_SIDECAR = Path(__file__).resolve().with_name("fake_sidecar.py")


def _chunk(idx: int, source: str, header: str, text: str) -> Chunk:
    return Chunk(
        id=f"{source}#{idx}",
        file_sha256=f"f{idx}",
        chunk_sha256=f"c{idx}",
        position=idx,
        page=None,
        section=header,
        file_date="2026-11-18",
        source=source,
        header=header,
        text=text,
    )


def _hash_client(tmp_path: Path, cache_dir: Path | None = None) -> SidecarClient:
    return SidecarClient(
        python=sys.executable,
        script=REAL_SIDECAR,
        cache_dir=cache_dir or (tmp_path / ".rag_cache"),
        model=DEFAULT_MODEL,
        model_dir=tmp_path / "models",
        threads=1,
        backend="hash",
        start_timeout=10.0,
        request_timeout=10.0,
    )


def _fake_client(
    tmp_path: Path,
    *,
    python: str | None = None,
    script: Path | None = None,
    backend: str = "ok",
    start_timeout: float = 1.0,
    request_timeout: float = 0.5,
) -> SidecarClient:
    return SidecarClient(
        python=python or sys.executable,
        script=FAKE_SIDECAR if script is None else script,
        cache_dir=tmp_path / "cache",
        model="fake/model",
        model_dir=tmp_path / "models",
        threads=1,
        backend=backend,
        start_timeout=start_timeout,
        request_timeout=request_timeout,
    )


def _set_fake_behavior(tmp_path: Path, behavior: dict[str, str]) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "fake_behavior.json").write_text(json.dumps(behavior), encoding="utf-8")


class _StubClient:
    """In-memory stub client for deterministic RRF and failure-count tests."""

    def __init__(
        self,
        dense_ranking: list[tuple[int, float]] | None = None,
        *,
        model: str = "stub/model",
    ) -> None:
        self._model = model
        self._dense_ranking = dense_ranking or []
        self.check_calls = 0
        self.start_calls = 0
        self.embed_calls = 0
        self.score_calls = 0
        self.close_calls = 0
        self.score_outcomes: list[list[tuple[int, float]] | Exception] = []

    def check(self) -> str | None:
        self.check_calls += 1
        return None

    def start(self) -> None:
        self.start_calls += 1

    def embed_passages(self, keys: list[str], texts: list[str]) -> list[str]:
        self.embed_calls += 1
        return list(keys)

    def score_query(self, query: str, keys: list[str]) -> list[tuple[int, float]]:
        self.score_calls += 1
        if self.score_outcomes:
            outcome = self.score_outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return list(self._dense_ranking)

    def close(self) -> None:
        self.close_calls += 1


class _StubBm25:
    def __init__(self, ranking: list[tuple[int, float]]) -> None:
        self._ranking = ranking

    def build(self, chunks: list[Chunk]) -> None:
        pass

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        return self._ranking[:k]


def test_resolve_mode_precedence_and_validation() -> None:
    assert resolve_mode(None, {}) == "auto"
    assert resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": ""}) == "auto"
    assert resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "bm25"}) == "bm25"
    assert resolve_mode("hybrid", {"VOICE_AGENT_RAG_RETRIEVAL": "bm25"}) == "hybrid"
    assert resolve_mode("BM25", {}) == "bm25"

    with pytest.raises(ValueError, match="bm25.*hybrid.*auto"):
        resolve_mode("invalid", {})
    with pytest.raises(ValueError, match="bm25.*hybrid.*auto"):
        resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "dense"})


def test_rrf_math_ties_and_late_component_full_ranking(tmp_path: Path) -> None:
    chunks = [
        _chunk(0, "a.md", "a.md", "alpha beta"),
        _chunk(1, "b.md", "b.md", "gamma delta"),
        _chunk(2, "c.md", "c.md", "epsilon zeta"),
        _chunk(3, "d.md", "d.md", "eta theta"),
    ]
    # BM25 returns only overlapping chunks (2, 0); dense ranks ALL 4 chunks (1, 2, 0, 3).
    stub_bm25 = _StubBm25([(2, 4.0), (0, 2.0)])
    stub_client = _StubClient([(1, 10.0), (2, 9.0), (0, 8.0), (3, 1.0)])
    cache = EmbedCache(tmp_path / "cache", "stub--model")

    retriever = HybridRetriever(
        stub_client,  # type: ignore[arg-type]
        cache,
        mode="hybrid",
        rrf_k=60,
        bm25=stub_bm25,  # type: ignore[arg-type]
        background=False,
    )
    try:
        retriever.build(chunks)
        assert retriever.ready
        ranked = retriever.rank("query", 10)
        expected = {
            2: 1.0 / 61 + 1.0 / 62,
            0: 1.0 / 62 + 1.0 / 63,
            1: 1.0 / 61,
            3: 1.0 / 64,
        }
        assert [idx for idx, _ in ranked] == [2, 0, 1, 3]
        for idx, score in ranked:
            assert score == pytest.approx(expected[idx])

        # Tie-breaking test: idx 2 at BM25 rank 1 + dense rank 2 vs idx 0 at BM25 rank 2 +
        # dense rank 1 -> identical fused score (1/61 + 1/62), so index 0 precedes index 2.
        stub_bm25._ranking = [(2, 5.0), (0, 4.0)]
        stub_client._dense_ranking = [(0, 9.0), (2, 8.0), (1, 7.0), (3, 6.0)]
        tied = retriever.rank("exact tie", 4)
        assert [idx for idx, _ in tied] == [0, 2, 1, 3]
        assert tied[0][1] == pytest.approx(tied[1][1])

        assert retriever.rank("", 4) == []
        assert retriever.rank("   ", 4) == []
        assert retriever.rank("exact tie", 0) == []
    finally:
        retriever.close()

    empty_retriever = HybridRetriever(
        _StubClient([]),  # type: ignore[arg-type]
        cache,
        mode="hybrid",
        background=False,
    )
    try:
        empty_retriever.build([])
        assert empty_retriever.rank("anything", 4) == []
    finally:
        empty_retriever.close()


def test_real_sidecar_hash_backend_interop_cache_reuse_and_docindex(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "solar.md").write_text(
        "# Solar Power\n\nSolar panels generate electricity in Cebu during sunny afternoons.\n",
        encoding="utf-8",
    )
    (corpus / "wind.md").write_text(
        "# Wind Power\n\nWind turbines spin steadily along coastal ridges in northern Luzon.\n",
        encoding="utf-8",
    )
    (corpus / "dup_a.md").write_text(
        "# Section Alpha\n\nShared passage body text with more than sixty characters in total.\n",
        encoding="utf-8",
    )
    (corpus / "dup_b.md").write_text(
        "# Section Beta\n\nShared passage body text with more than sixty characters in total.\n",
        encoding="utf-8",
    )

    cache_dir = corpus / ".rag_cache"
    slug = model_slug(DEFAULT_MODEL)
    cache = EmbedCache(cache_dir, slug)

    # First index populates the on-disk cache.
    client1 = _hash_client(tmp_path, cache_dir=cache_dir)
    embed_calls_1: list[list[str]] = []
    orig_embed_1 = client1.embed_passages

    def tracked_embed_1(keys: list[str], texts: list[str]) -> list[str]:
        embed_calls_1.append(list(keys))
        return orig_embed_1(keys, texts)

    client1.embed_passages = tracked_embed_1  # type: ignore[method-assign]
    r1 = HybridRetriever(client1, cache, mode="hybrid", background=False)
    try:
        idx = DocIndex(corpus, retriever_factory=lambda: r1)
        idx.refresh()
        assert r1.wait_ready(timeout=5.0)
        assert r1.status() == f"hybrid:{DEFAULT_MODEL}"

        # Hidden .rag_cache is ignored by DocIndex._walk_folder.
        assert all(
            rec.name != ".rag_cache" and ".rag_cache" not in rec.path for rec in idx.records()
        )
        assert idx.failures() == []

        # Same text under two headers gets two distinct keys.
        dup_chunks = [c for c in idx.chunks if c.source in ("dup_a.md", "dup_b.md")]
        assert len(dup_chunks) == 2
        dup_keys = [embed_key(c.header, c.text) for c in dup_chunks]
        assert dup_keys[0] != dup_keys[1]
        assert all(cache.path_for(k).is_file() for k in dup_keys)

        hits = idx.search("Where do solar panels generate electricity in Cebu?", k=4)
        assert hits
        assert hits[0]["source"] == "solar.md"
        assert len(embed_calls_1) == 1
        assert len(embed_calls_1[0]) == len(idx.chunks)
    finally:
        r1.close()

    mtimes_before = {p.name: p.stat().st_mtime_ns for p in (cache_dir / slug).glob("*.npy")}

    # Second HybridRetriever over the same corpus reuses cached vectors and embeds nothing.
    client2 = _hash_client(tmp_path, cache_dir=cache_dir)
    embed_calls_2: list[list[str]] = []

    def tracked_embed_2(keys: list[str], texts: list[str]) -> list[str]:
        embed_calls_2.append(list(keys))
        return client2.__class__.embed_passages(client2, keys, texts)

    client2.embed_passages = tracked_embed_2  # type: ignore[method-assign]
    r2 = HybridRetriever(client2, cache, mode="hybrid", background=False)
    try:
        idx2 = DocIndex(corpus, retriever_factory=lambda: r2)
        idx2.refresh()
        assert r2.wait_ready(timeout=5.0)
        assert embed_calls_2 == []
        mtimes_after = {p.name: p.stat().st_mtime_ns for p in (cache_dir / slug).glob("*.npy")}
        assert mtimes_after == mtimes_before

        # Changing one file re-embeds only that file's key and prunes the old key.
        solar_path = corpus / "solar.md"
        old_solar_chunk = next(c for c in idx2.chunks if c.source == "solar.md")
        old_solar_key = embed_key(old_solar_chunk.header, old_solar_chunk.text)
        solar_path.write_text(
            "# Solar Power\n\n"
            "Solar panels now power coastal desalination plants across southern Bohol.\n",
            encoding="utf-8",
        )
        st = solar_path.stat()
        os.utime(solar_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        idx2.refresh()
        assert r2.wait_ready(timeout=5.0)
        assert len(embed_calls_2) == 1
        assert len(embed_calls_2[0]) == 1
        assert not cache.path_for(old_solar_key).exists()
        hits_after = idx2.search("desalination plants Bohol", k=4)
        assert hits_after[0]["source"] == "solar.md"
    finally:
        r2.close()


def test_bm25_serves_while_warming_and_stale_generation_stops(tmp_path: Path) -> None:
    # Build >16 chunks so embed_passages runs at least 2 batches (16 + remainder).
    chunks = [
        _chunk(i, f"f{i}.md", f"f{i}.md", f"passage number {i} about topic{i}") for i in range(20)
    ]
    client = _fake_client(tmp_path)
    cache = EmbedCache(tmp_path / "cache", "fake--model")

    batch1_entered = threading.Event()
    release_batch1 = threading.Event()
    embedded_batches: list[list[str]] = []
    orig_embed = client.embed_passages

    def gated_embed(keys: list[str], texts: list[str]) -> list[str]:
        embedded_batches.append(list(keys))
        if len(embedded_batches) == 1:
            batch1_entered.set()
            assert release_batch1.wait(timeout=5.0)
        return orig_embed(keys, texts)

    client.embed_passages = gated_embed  # type: ignore[method-assign]
    retriever = HybridRetriever(client, cache, mode="hybrid", background=True)
    try:
        retriever.build(chunks)
        assert batch1_entered.wait(timeout=5.0)

        # While warming: not ready, status shows warming, rank returns BM25 immediately.
        assert not retriever.ready
        assert retriever.status().startswith("bm25 (hybrid warming:")
        bm25_only = Bm25Retriever()
        bm25_only.build(chunks)
        assert retriever.rank("topic5", 3) == bm25_only.rank("topic5", 3)

        # Trigger a new generation while gen 1 is blocked after batch 1.
        new_chunks = [_chunk(0, "new.md", "new.md", "fresh replacement passage zebra")]
        retriever.build(new_chunks)
        release_batch1.set()

        assert retriever.wait_ready(timeout=5.0)
        assert retriever.ready
        assert retriever.status() == "hybrid:fake/model"
        # Gen 1 had 20 keys (would have sent batch 2 of 4 keys) -> stopped after batch 1;
        # Gen 2 sent 1 batch of 1 key. Total batches == 2 (not 3!).
        assert [len(b) for b in embedded_batches] == [16, 1]
        assert retriever.rank("zebra", 3)[0][0] == 0
    finally:
        release_batch1.set()
        retriever.close()


@pytest.mark.parametrize(
    ("scenario", "mode"),
    [
        ("missing_python", "auto"),
        ("missing_python", "hybrid"),
        ("missing_script", "auto"),
        ("fail_check", "hybrid"),
        ("hang_embed", "auto"),
        ("eof_embed", "hybrid"),
        ("garbage_embed", "hybrid"),
        ("bad_done_embed", "hybrid"),
    ],
)
def test_build_fallbacks_serve_bm25_never_raise_and_log_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
    mode: str,
) -> None:
    chunks = [
        _chunk(0, "a.md", "a.md", "alpha bravo charlie delta"),
        _chunk(1, "b.md", "b.md", "echo foxtrot golf hotel"),
    ]
    bm25_ref = Bm25Retriever()
    bm25_ref.build(chunks)
    expected = bm25_ref.rank("bravo", 2)

    cache = EmbedCache(tmp_path / "cache", "fake--model")
    if scenario == "missing_python":
        client = _fake_client(tmp_path, python=str(tmp_path / "no-such-python"))
    elif scenario == "missing_script":
        client = _fake_client(tmp_path, script=tmp_path / "no-such-sidecar.py")
    elif scenario == "fail_check":
        client = _fake_client(tmp_path, backend="fail_check")
    elif scenario == "hang_embed":
        _set_fake_behavior(tmp_path, {"embed_passages": "hang"})
        client = _fake_client(tmp_path, request_timeout=0.15)
    elif scenario == "eof_embed":
        _set_fake_behavior(tmp_path, {"embed_passages": "eof"})
        client = _fake_client(tmp_path)
    elif scenario == "garbage_embed":
        _set_fake_behavior(tmp_path, {"embed_passages": "garbage"})
        client = _fake_client(tmp_path)
    else:
        client = _fake_client(tmp_path)
        client.embed_passages = lambda keys, texts: []  # type: ignore[method-assign]

    retriever = HybridRetriever(client, cache, mode=mode, background=False)  # type: ignore[arg-type]
    try:
        retriever.build(chunks)
        assert not retriever.ready
        assert retriever.rank("bravo", 2) == expected
        if mode == "hybrid":
            assert retriever.status().startswith("bm25 (hybrid unavailable:")
        else:
            assert retriever.status() == "bm25"

        # A second build keeps dense disabled and does not log a duplicate line.
        retriever.build(chunks)
        assert retriever.rank("bravo", 2) == expected
    finally:
        retriever.close()

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert err_lines[0].startswith("rag: hybrid unavailable (")
    assert err_lines[0].endswith("); serving BM25")
    assert not client.alive


@pytest.mark.parametrize("failure_mode", ["eof", "garbage", "hang", "ok_false"])
def test_score_query_fallbacks_and_two_consecutive_failures_trip_breaker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure_mode: str,
) -> None:
    chunks = [
        _chunk(0, "a.md", "a.md", "alpha bravo charlie"),
        _chunk(1, "b.md", "b.md", "delta echo foxtrot"),
    ]
    bm25_ref = Bm25Retriever()
    bm25_ref.build(chunks)
    expected = bm25_ref.rank("bravo", 2)

    cache = EmbedCache(tmp_path / "cache", "fake--model")
    client = _fake_client(tmp_path, request_timeout=0.15)
    retriever = HybridRetriever(client, cache, mode="hybrid", background=False)
    try:
        retriever.build(chunks)
        assert retriever.ready
        assert client.alive

        # Failure 1: falls back to BM25, sidecar not yet marked permanently broken.
        _set_fake_behavior(tmp_path, {"score_query": failure_mode})
        assert retriever.rank("bravo", 2) == expected
        assert retriever.ready

        if failure_mode == "ok_false":
            # Sidecar stayed alive; a subsequent success resets the consecutive counter.
            _set_fake_behavior(tmp_path, {"score_query": "ok"})
            assert retriever.rank("bravo", 2)
            assert retriever.ready
            _set_fake_behavior(tmp_path, {"score_query": failure_mode})
            assert retriever.rank("bravo", 2) == expected
            assert retriever.ready

        # Second consecutive failure marks dense broken, logs ONE line, and closes the sidecar.
        assert retriever.rank("bravo", 2) == expected
        assert not retriever.ready
        assert not client.alive
        assert retriever.status().startswith("bm25 (hybrid unavailable:")
        assert retriever.rank("bravo", 2) == expected
    finally:
        retriever.close()

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert err_lines[0].startswith("rag: hybrid unavailable (")


def test_readonly_cache_dir_falls_back_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks = [_chunk(0, "a.md", "a.md", "readonly cache directory fallback test")]
    ro_dir = tmp_path / "readonly_cache"
    ro_dir.mkdir()
    ro_dir.chmod(0o555)
    try:
        client = _hash_client(tmp_path, cache_dir=ro_dir)
        cache = EmbedCache(ro_dir, model_slug(DEFAULT_MODEL))
        retriever = HybridRetriever(client, cache, mode="hybrid", background=False)
        try:
            retriever.build(chunks)
            assert not retriever.ready
            assert retriever.status().startswith("bm25 (hybrid unavailable:")
            assert retriever.rank("readonly", 2) == [
                (0, pytest.approx(retriever.rank("readonly", 2)[0][1]))
            ]
        finally:
            retriever.close()
    finally:
        ro_dir.chmod(0o755)

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert err_lines[0].startswith("rag: hybrid unavailable (")


def test_bm25_mode_never_spawns_and_factory_reuses_one_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[Any] = []
    orig_popen = subprocess.Popen

    def spy_popen(*args: Any, **kwargs: Any) -> Any:
        popen_calls.append(args[0] if args else kwargs)
        return orig_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy_popen)

    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# Doc\n\nAlpha beta gamma delta epsilon zeta eta theta iota kappa lambda.\n",
        encoding="utf-8",
    )

    bm25_factory = make_retriever_factory(mode="bm25", cache_dir=corpus / ".rag_cache")
    assert bm25_factory is Bm25Retriever
    idx_bm25 = DocIndex(corpus, retriever_factory=bm25_factory)
    idx_bm25.refresh()
    assert idx_bm25.search("alpha", 2)
    assert popen_calls == []

    # Hybrid/auto factory returns the SAME HybridRetriever instance across DocIndex.refresh() calls.
    hybrid_factory = make_retriever_factory(
        mode="hybrid",
        cache_dir=corpus / ".rag_cache",
        embed_python=sys.executable,
        script=REAL_SIDECAR,
        model=DEFAULT_MODEL,
        background=False,
    )
    first_instance = hybrid_factory()
    assert isinstance(first_instance, HybridRetriever)
    assert hybrid_factory() is first_instance
    # Patch SidecarClient._backend to "hash" for offline testing without touching fastembed.
    assert first_instance._client is not None
    first_instance._client._backend = "hash"
    try:
        idx = DocIndex(corpus, retriever_factory=hybrid_factory)
        idx.refresh()
        serve_spawns = [c for c in popen_calls if "--check" not in c]
        assert len(serve_spawns) == 1
        total_spawns_before = len(popen_calls)

        # Refresh after file change keeps the SAME sidecar process alive (no second Popen).
        doc = corpus / "a.md"
        doc.write_text(
            "# Doc\n\nUpdated text with plenty of words for chunking and retrieval verification.\n",
            encoding="utf-8",
        )
        st = doc.stat()
        os.utime(doc, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        idx.refresh()
        assert len(popen_calls) == total_spawns_before
        assert idx.search("Updated text", 2)
    finally:
        first_instance.close()


def test_metrics_cli_forced_hybrid_unavailable_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(
        "# Notes\n\nSolar panels produce clean power.\n", encoding="utf-8"
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "q1",
                "q": "What produces clean power?",
                "type": "lexical",
                "relevant": [{"file": "doc.md", "contains": "Solar panels produce clean power."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("VOICE_AGENT_EMBED_PYTHON", str(tmp_path / "missing-python"))
    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--retriever",
            "hybrid",
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert rc != 0
    captured = capsys.readouterr()
    assert "bm25 (hybrid unavailable:" in captured.err
    assert not (corpus / ".rag_cache").exists()
    assert not cache_dir.exists()

    rc_bm25 = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--retriever",
            "bm25",
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert rc_bm25 == 0
    captured_bm25 = capsys.readouterr()
    assert "retriever: bm25" in captured_bm25.out
    assert not (corpus / ".rag_cache").exists()
    assert not cache_dir.exists()
    assert "fastembed" not in sys.modules


def test_metrics_cli_cache_dir_default_and_never_writes_into_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = metrics._parse_args([])
    assert parsed.cache_dir == metrics.DEFAULT_CACHE_DIR
    assert parsed.cache_dir.parts[-3:] == (".cache", "voice_agent", "rag_metrics")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(
        "# Notes\n\nSolar panels produce clean power.\n", encoding="utf-8"
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "q1",
                "q": "What produces clean power?",
                "type": "lexical",
                "relevant": [{"file": "doc.md", "contains": "Solar panels produce clean power."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_cache_dirs: list[Path | str | None] = []
    cache_dir = tmp_path / "custom_cache"

    def spy_factory(
        *,
        mode: str = "auto",
        cache_dir: Path | str | None = None,
        background: bool = True,
    ) -> Any:
        seen_cache_dirs.append(cache_dir)
        return Bm25Retriever

    monkeypatch.setattr(metrics, "make_retriever_factory", spy_factory)
    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--retriever",
            "auto",
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert rc == 0
    assert seen_cache_dirs == [cache_dir]
    assert not (corpus / ".rag_cache").exists()
