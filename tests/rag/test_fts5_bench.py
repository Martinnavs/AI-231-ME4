"""Offline 5000-chunk synthetic benchmark for Fts5Retriever vs Bm25Retriever."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CHUNK_COUNT = 5000
WORDS_PER_CHUNK = 85  # ~85 * 7 - 1 = 594 chars per chunk (~600 chars)
VOCAB_SIZE = 5000
QUERY_COUNT = 200
QUERY_WORDS = 3
SEED = 20261119

# FD6 bounds (from PLAN-fts5.md / ticket fts5-F06-bench.md)
MAX_REOPEN_MS = 250.0
MAX_QUERY_P50_MS = 50.0
MAX_RSS_MB = 150.0
MAX_DB_MB = 40.0
MAX_BUILD_S = 10.0

_WORKER_SCRIPT = r"""
from __future__ import annotations

import bisect
import json
import random
import resource
import statistics
import sys
import time
from pathlib import Path

from voice_agent.rag.docindex import Bm25Retriever, Chunk
from voice_agent.rag.fts5 import Fts5Retriever


def _peak_rss_mb() -> float:
    # ru_maxrss is inherited across fork/exec (a big parent test process inflates it);
    # VmHWM is the child's own high-water mark.
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


CHUNK_COUNT = 5000
WORDS_PER_CHUNK = 85
VOCAB_SIZE = 5000
QUERY_COUNT = 200
QUERY_WORDS = 3
SEED = 20261119


def _build_zipf_vocab_and_cdf(vocab_size: int) -> tuple[list[str], list[float]]:
    consonants = "bcdfghjklmnpqrstvwxz"
    vowels = "aeiou"
    syllables = [c + v for c in consonants for v in vowels]  # 100 syllables
    vocab: list[str] = []
    for i in range(vocab_size):
        s0 = syllables[i % len(syllables)]
        s1 = syllables[(i // len(syllables)) % len(syllables)]
        s2 = syllables[(i // (len(syllables) * len(syllables))) % len(syllables)]
        vocab.append(s0 + s1 + s2)

    weights = [1.0 / (rank + 1) for rank in range(vocab_size)]
    total = sum(weights)
    cdf: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    cdf[-1] = 1.0
    return vocab, cdf


def _generate_corpus(
    seed: int,
) -> tuple[list[Chunk], list[str], float]:
    rng = random.Random(seed)
    vocab, cdf = _build_zipf_vocab_and_cdf(VOCAB_SIZE)

    chunks: list[Chunk] = []
    total_chars = 0
    for idx in range(CHUNK_COUNT):
        file_idx = idx // 10
        pos = idx % 10
        words = [
            vocab[bisect.bisect_left(cdf, rng.random())]
            for _ in range(WORDS_PER_CHUNK)
        ]
        text = " ".join(words)
        total_chars += len(text)
        source = f"doc_{file_idx:03d}.md"
        header = f"{source} > Section {pos}"
        chunks.append(
            Chunk(
                id=f"{source}#{pos}",
                file_sha256=f"fsha-{file_idx:03d}",
                chunk_sha256=f"csha-{idx:05d}",
                position=pos,
                page=None,
                section=f"Section {pos}",
                file_date="2026-01-15",
                source=source,
                header=header,
                text=text,
            )
        )

    queries: list[str] = []
    for _ in range(QUERY_COUNT):
        q_words = [
            vocab[bisect.bisect_left(cdf, rng.random())]
            for _ in range(QUERY_WORDS)
        ]
        queries.append(" ".join(q_words))

    avg_chars = total_chars / len(chunks)
    return chunks, queries, avg_chars


def _bench_fts5(
    mode: str,
    db_path: Path,
    root: Path,
    chunks: list[Chunk],
    queries: list[str],
) -> dict[str, float | int | str]:
    tok = "pretok" if mode == "fts5-pretok" else "porter"
    t0 = time.perf_counter()
    retriever = Fts5Retriever(db_path, root, tokenizer=tok)  # type: ignore[arg-type]
    retriever.build(chunks)
    build_s = time.perf_counter() - t0
    status_str = retriever.status()
    retriever.close()

    db_bytes = db_path.stat().st_size

    t1 = time.perf_counter()
    reopened = Fts5Retriever(db_path, root, tokenizer=tok)  # type: ignore[arg-type]
    reopened.build(chunks)
    first_hits = reopened.rank(queries[0], len(chunks))
    reopen_ms = (time.perf_counter() - t1) * 1000.0
    sync_stats = reopened.last_sync_stats

    latencies_ms: list[float] = []
    nonempty = 0
    for q in queries:
        qs = time.perf_counter()
        hits = reopened.rank(q, len(chunks))
        latencies_ms.append((time.perf_counter() - qs) * 1000.0)
        if hits:
            nonempty += 1
    reopened.close()

    peak_rss_mb = _peak_rss_mb()
    return {
        "backend": mode,
        "status": status_str,
        "build_s": build_s,
        "reopen_ms": reopen_ms,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))],
        "peak_rss_mb": peak_rss_mb,
        "db_mb": db_bytes / (1024.0 * 1024.0),
        "first_hits": len(first_hits),
        "nonempty_queries": nonempty,
        "reopen_inserted": sync_stats.inserted if sync_stats else -1,
        "reopen_deleted": sync_stats.deleted if sync_stats else -1,
    }


def _bench_bm25(
    chunks: list[Chunk],
    queries: list[str],
) -> dict[str, float | int | str]:
    t0 = time.perf_counter()
    retriever = Bm25Retriever()
    retriever.build(chunks)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    rebuilt = Bm25Retriever()
    rebuilt.build(chunks)
    first_hits = rebuilt.rank(queries[0], len(chunks))
    reopen_ms = (time.perf_counter() - t1) * 1000.0

    latencies_ms: list[float] = []
    nonempty = 0
    for q in queries:
        qs = time.perf_counter()
        hits = rebuilt.rank(q, len(chunks))
        latencies_ms.append((time.perf_counter() - qs) * 1000.0)
        if hits:
            nonempty += 1

    peak_rss_mb = _peak_rss_mb()
    return {
        "backend": "rank-bm25",
        "status": "bm25",
        "build_s": build_s,
        "reopen_ms": reopen_ms,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))],
        "peak_rss_mb": peak_rss_mb,
        "db_mb": 0.0,
        "first_hits": len(first_hits),
        "nonempty_queries": nonempty,
        "reopen_inserted": 0,
        "reopen_deleted": 0,
    }


def main() -> None:
    mode = sys.argv[1]
    root = Path(sys.argv[2])
    chunks, queries, avg_chars = _generate_corpus(SEED)
    if mode in ("fts5-pretok", "fts5-porter"):
        db_path = root / ".rag_cache" / "fts5" / f"{mode}.sqlite3"
        result = _bench_fts5(mode, db_path, root, chunks, queries)
    elif mode == "rank-bm25":
        result = _bench_bm25(chunks, queries)
    else:
        raise ValueError(f"unknown mode: {mode}")
    result["avg_chars"] = avg_chars
    print(json.dumps(result))


if __name__ == "__main__":
    main()
"""


def _run_backend_subprocess(mode: str, workdir: Path) -> dict[str, Any]:
    env = dict(os.environ)
    src_dir = str(REPO_ROOT / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}{os.pathsep}{existing_pp}"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, mode, str(workdir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{mode} subprocess failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"{mode} produced no output; stderr={proc.stderr}"
    return dict(json.loads(lines[-1]))


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'backend':<12} | {'build_s':>7} | {'reopen_ms':>9} | "
        f"{'p50_ms':>7} | {'p95_ms':>7} | {'rss_mb':>7} | {'db_mb':>6}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{str(row['backend']):<12} | "
            f"{float(row['build_s']):>7.3f} | "
            f"{float(row['reopen_ms']):>9.2f} | "
            f"{float(row['p50_ms']):>7.2f} | "
            f"{float(row['p95_ms']):>7.2f} | "
            f"{float(row['peak_rss_mb']):>7.2f} | "
            f"{float(row['db_mb']):>6.2f}"
        )
    return "\n".join(lines)


def test_fts5_5000_chunk_bench(tmp_path: Path) -> None:
    """Measure build, reopen-to-first-query, p50, peak RSS, and DB size at 5000 chunks."""
    rows: list[dict[str, Any]] = []
    for mode in ("fts5-pretok", "fts5-porter", "rank-bm25"):
        mode_dir = tmp_path / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        rows.append(_run_backend_subprocess(mode, mode_dir))

    print("\n" + _format_table(rows))

    by_mode = {str(r["backend"]): r for r in rows}
    bm25 = by_mode["rank-bm25"]
    bm25_rss = float(bm25["peak_rss_mb"])

    for mode in ("fts5-pretok", "fts5-porter"):
        row = by_mode[mode]
        assert row["status"] == "fts5:file", f"{mode} fell back: {row['status']}"
        assert row["reopen_inserted"] == 0 and row["reopen_deleted"] == 0
        assert int(row["nonempty_queries"]) > 0
        assert float(row["reopen_ms"]) <= MAX_REOPEN_MS, (
            f"{mode} reopen_ms={row['reopen_ms']:.2f} > {MAX_REOPEN_MS}"
        )
        assert float(row["p50_ms"]) <= MAX_QUERY_P50_MS, (
            f"{mode} p50_ms={row['p50_ms']:.2f} > {MAX_QUERY_P50_MS}"
        )
        assert float(row["peak_rss_mb"]) <= MAX_RSS_MB, (
            f"{mode} peak_rss_mb={row['peak_rss_mb']:.2f} > {MAX_RSS_MB}"
        )
        assert float(row["peak_rss_mb"]) <= bm25_rss, (
            f"{mode} peak_rss_mb={row['peak_rss_mb']:.2f} > rank-bm25 {bm25_rss:.2f}"
        )
        assert float(row["db_mb"]) <= MAX_DB_MB, f"{mode} db_mb={row['db_mb']:.2f} > {MAX_DB_MB}"
        assert float(row["build_s"]) <= MAX_BUILD_S, (
            f"{mode} build_s={row['build_s']:.3f} > {MAX_BUILD_S}"
        )
