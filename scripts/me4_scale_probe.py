"""Scale probe for ME4: index a corpus in background, sample system metrics, then search.

Runs DocIndex.refresh() in a thread and samples RSS, cache size, disk, memory once
per second. After index is ready: search latency p50/p95, needle recall at k, reopen
time, and first-searchable vs ready timestamps.  Linux only, read-only on corpus.

Usage:
    uv run python scripts/me4_scale_probe.py --corpus DIR --needles needles.json [--warm]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from voice_agent.rag.backends import make_retriever_factory
from voice_agent.rag.docindex import DocIndex

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo -> {key: kB_value}."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # kB
    except OSError:
        pass
    return info


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _rss_current_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return 0.0


def _cache_size_mb(cache_dir: Path) -> float:
    total = 0
    if cache_dir.exists():
        for f in cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / 1_000_000


def _free_disk_mb(path: Path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize // 1_000_000


def _sample_metrics(cache_dir: Path, corpus: Path) -> dict:
    meminfo = _read_meminfo()
    memavail = meminfo.get("MemAvailable", 0) / 1024  # kB -> MB
    swap_total = meminfo.get("SwapTotal", 0) / 1024
    swap_free = meminfo.get("SwapFree", 0) / 1024
    return {
        "s": round(time.perf_counter(), 2),
        "rss_mb": round(_rss_current_mb(), 1),
        "peak_rss_mb": round(_rss_mb(), 1),
        "cache_mb": round(_cache_size_mb(cache_dir), 1),
        "disk_free_mb": _free_disk_mb(corpus),
        "memavail_mb": round(memavail, 1),
        "swap_used_mb": round(swap_total - swap_free, 1),
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="ME4 scale probe")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--needles", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sample-s", type=float, default=1.0, dest="sample_s")
    ap.add_argument("--samples", type=Path, default=None)
    ap.add_argument("--warm", action="store_true", help="Keep existing .rag_cache (warm run)")
    args = ap.parse_args()

    cache_dir = args.corpus / ".rag_cache"
    needles: list[dict] = json.loads(args.needles.read_text())

    # Cache policy: cold by default
    cache_mode = "warm" if args.warm else "cold"
    if cache_mode == "cold" and cache_dir.exists():
        shutil.rmtree(str(cache_dir))

    # Build DocIndex with FTS5 (the probe mode used in synth_check.py)
    factory = make_retriever_factory(mode="fts5", cache_dir=cache_dir, embed_python=None)
    ix = DocIndex(args.corpus, factory)

    # Sampling loop
    samples: list[dict] = []
    files_count = 0
    first_searchable_s: float | None = None
    ready = threading.Event()

    def _sampler() -> None:
        nonlocal files_count, first_searchable_s
        t0 = time.perf_counter()
        while not ready.is_set():
            with contextlib.suppress(Exception):
                files_count = len(ix.records())
            rec = _sample_metrics(cache_dir, args.corpus)
            rec["files"] = files_count
            samples.append(rec)
            if first_searchable_s is None and files_count > 0:
                first_searchable_s = time.perf_counter() - t0
            time.sleep(args.sample_s)

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()

    # Index
    t0 = time.perf_counter()
    ix.refresh()
    ready_s = time.perf_counter() - t0
    ready.set()

    # Wait a moment for last sample, then record one more measurement
    time.sleep(0.2)
    with contextlib.suppress(Exception):
        files_count = len(ix.records())
    # If sampler never saw files > 0 before refresh ended, set to ready_s
    if first_searchable_s is None:
        first_searchable_s = ready_s if files_count > 0 else 0.0

    # Final metrics
    final_rss = _rss_mb()
    cache_mb = _cache_size_mb(cache_dir)

    # Search latency + recall
    latencies: list[float] = []
    hits = 0
    for n in needles:
        t1 = time.perf_counter()
        results = ix.search(n["query"], args.k)
        latencies.append((time.perf_counter() - t1) * 1000)
        texts = " ".join(r.get("text", "") for r in results)
        if n["answer"] in texts or n["query"] in texts:
            hits += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    recall = hits / len(needles) if needles else 0

    # Reopen test (always warm: cache still exists)
    reopen_factory = make_retriever_factory(mode="fts5", cache_dir=cache_dir, embed_python=None)
    ix2 = DocIndex(args.corpus, reopen_factory)
    t_reopen = time.perf_counter()
    ix2.refresh()
    reopen_s = time.perf_counter() - t_reopen
    # First query after reopen
    if needles:
        t_query = time.perf_counter()
        ix2.search(needles[0]["query"], args.k)
        reopen_first_query_s = time.perf_counter() - t_query
    else:
        reopen_first_query_s = 0

    # Build result
    result = {
        "cache": cache_mode,
        "files": len(needles),
        "ready_s": round(ready_s, 2),
        "first_searchable_s": round(first_searchable_s, 3),
        "chunks": len(ix.chunks),
        "peak_rss_mb": round(final_rss),
        "cache_mb": round(cache_mb),
        "search_p50_ms": round(p50, 2),
        "search_p95_ms": round(p95, 2),
        "needle_recall": f"{hits}/{len(needles)}",
        "needle_recall_pct": round(recall * 100, 1),
        "reopen_s": round(reopen_s, 2),
        "reopen_first_query_ms": round(reopen_first_query_s * 1000, 2),
        "failures": len(ix.failures()),
        "samples": len(samples),
    }

    print(json.dumps(result, indent=2))

    # Write TSV samples
    if args.samples:
        _write_tsv(args.samples, samples)


def _write_tsv(path: Path, samples: Sequence[dict]) -> None:
    cols = [
        "s",
        "files",
        "rss_mb",
        "peak_rss_mb",
        "cache_mb",
        "disk_free_mb",
        "memavail_mb",
        "swap_used_mb",
    ]
    lines = ["\t".join(cols)]
    for s in samples:
        lines.append("\t".join(str(s.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
