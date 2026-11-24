"""Warm worker processes for ingest parsing with crash/timeout/MEMORY handling."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import resource
import time
from multiprocessing.connection import Connection, Pipe, wait
from pathlib import Path
from typing import Any

from voice_agent.rag.canonical_cache import CanonicalCache
from voice_agent.rag.docindex import LoadedFile, load_file


def run_worker(
    conn: Connection,
    converters: dict[str, str],
    cache_dir: str,
    src_str: str,
    worker_mem_mb: int,
    loader: Any = None,
) -> None:
    if worker_mem_mb > 0:
        resource.setrlimit(resource.RLIMIT_AS, (worker_mem_mb << 20, worker_mem_mb << 20))
    cache = CanonicalCache(cache_dir, src_str)
    real = loader or load_file
    while True:
        try:
            path_str, rel_key = conn.recv()
        except EOFError:
            return
        lf = real(
            Path(path_str),
            rel_key,
            converters=converters,
            canonical_cache=cache,
            prev=None,
            now_iso=time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        )
        conn.send((rel_key, lf))


def _make_error(rel_key: str, status: str, error: str) -> LoadedFile:
    return LoadedFile(
        status=status,
        error=error,
        kind="unknown",
        size=0,
        mtime=0.0,
        mtime_ns=0,
        ctime_ns=0,
        sha256="",
        chunks=[],
        dates=[],
        canonical_key="",
        unchanged=False,
        note=None,
    )


def _start_worker(
    idx: int,
    w_procs: list[Any],
    w_ins: list[Any],
    w_outs: list[Any],
    ctx: Any,
    converters: dict[str, str],
    cache_dir: str,
    src: Path,
    worker_mem_mb: int,
    loader: Any,
    idle_workers: list[int],
) -> None:
    if w_ins[idx] is not None:
        w_ins[idx].close()
    if w_outs[idx] is not None:
        w_outs[idx].close()
    c1, c2 = Pipe(duplex=True)
    p = ctx.Process(
        target=run_worker,
        args=(c2, converters, cache_dir, str(src), worker_mem_mb, loader),
        daemon=True,
    )
    p.start()
    w_procs[idx] = p
    w_ins[idx] = c1
    w_outs[idx] = c1
    idle_workers.append(idx)


def _collect(
    *,
    w_outs: list[Any],
    w_ins: list[Any],
    w_procs: list[Any],
    jobs: dict[str, float],
    results: dict[str, LoadedFile],
    in_flight: list[int],
    file_timeout_s: int,
    timeout: float,
    ctx: Any,
    converters: dict[str, str],
    cache_dir: str,
    src: Path,
    worker_mem_mb: int,
    loader: Any,
    worker_to_job: dict[int, str],
    idle_workers: list[int],
) -> tuple[bool, bool]:
    """Collect results, check timeouts, reap dead workers.

    Args:
        timeout: seconds to wait on pipes (0 = non-blocking poll).

    Returns ``(found_result, had_timeout)``.
    """
    found = False

    # Wait for at least one pipe to have data (or timeout)
    if timeout > 0 and jobs:
        ready = [out for out in w_outs if out is not None]
        if ready:
            wait(ready, timeout=timeout)

    # Collect from ready pipes
    for i, out in enumerate(w_outs):
        if out.poll(0):
            try:
                rk, lf = out.recv()
                if rk in jobs:
                    results[rk] = lf
                    del jobs[rk]
                    in_flight[0] -= 1
                    found = True
                    idle_workers.append(i)
                    worker_to_job.pop(i, None)
            except (EOFError, OSError):
                crashed_rk = worker_to_job.pop(i, None)
                if crashed_rk and crashed_rk in jobs:
                    results[crashed_rk] = _make_error(
                        crashed_rk, "unreadable", "parser crashed (pipe closed)"
                    )
                    del jobs[crashed_rk]
                    in_flight[0] = max(0, in_flight[0] - 1)
                idle_workers.append(i)

    # Check for timed-out jobs
    timed_out = None
    if jobs and file_timeout_s > 0:
        now = time.monotonic()
        for rk, t in jobs.items():
            if now - t > file_timeout_s:
                timed_out = rk
                break

    if timed_out is not None:
        results[timed_out] = _make_error(
            timed_out, "unreadable", f"parser timed out after {file_timeout_s}s"
        )
        del jobs[timed_out]
        in_flight[0] = max(0, in_flight[0] - 1)
        timed_worker = None
        for widx, wrk in worker_to_job.items():
            if wrk == timed_out:
                timed_worker = widx
                break
        if timed_worker is not None:
            proc = w_procs[timed_worker]
            worker_to_job.pop(timed_worker, None)
            if proc is not None and proc.exitcode is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                proc.join(timeout=2)
            _start_worker(
                timed_worker,
                w_procs,
                w_ins,
                w_outs,
                ctx,
                converters,
                cache_dir,
                src,
                worker_mem_mb,
                loader,
                idle_workers,
            )
        return False, True

    # Reap dead workers
    crashed: list[int] = []
    for i, proc in enumerate(w_procs):
        if proc is not None and proc.exitcode is None:
            try:
                pid, status = os.waitpid(proc.pid, os.WNOHANG)
                if pid:
                    if os.WIFEXITED(status):
                        proc.exitcode = os.WEXITSTATUS(status)
                    elif os.WIFSIGNALED(status):
                        proc.exitcode = -os.WTERMSIG(status)
            except ChildProcessError:
                pass
            if proc.exitcode is not None:
                crashed.append(i)
    if crashed:
        for idx in crashed:
            rk = worker_to_job.pop(idx, None)
            if rk and rk in jobs:
                results[rk] = _make_error(rk, "unreadable", "parser crashed (child exited)")
                del jobs[rk]
                in_flight[0] = max(0, in_flight[0] - 1)
        for i in crashed:
            _start_worker(
                i,
                w_procs,
                w_ins,
                w_outs,
                ctx,
                converters,
                cache_dir,
                src,
                worker_mem_mb,
                loader,
                idle_workers,
            )
    return found, False


def supervisor_parse(
    files: list[tuple[Path, str]],
    converters: dict[str, str],
    cache_dir: str,
    src: Path,
    *,
    workers: int = 3,
    worker_mem_mb: int = 0,
    file_timeout_s: int = 300,
    loader: Any = None,
    window_check: Any = None,
) -> list[tuple[str, LoadedFile]]:
    """Dispatch ``files`` to ``workers`` warm processes, returns in sorted rel_key order."""
    if not files:
        return []

    dispatch_order = sorted(files, key=lambda f: f[0].stat().st_size)
    ctx = mp.get_context("fork")

    w_procs: list[Any] = [None] * workers
    w_ins: list[Any] = [None] * workers
    w_outs: list[Any] = [None] * workers

    worker_to_job: dict[int, str] = {}
    idle_workers: list[int] = []

    for i in range(workers):
        _start_worker(
            i,
            w_procs,
            w_ins,
            w_outs,
            ctx,
            converters,
            cache_dir,
            src,
            worker_mem_mb,
            loader,
            idle_workers,
        )

    jobs: dict[str, float] = {}
    results: dict[str, LoadedFile] = {}
    in_flight = [0]

    # Phase 1: eager dispatch (non-blocking collect first)
    for _fp, _rel_key in dispatch_order:
        _collect(
            w_outs=w_outs,
            w_ins=w_ins,
            w_procs=w_procs,
            jobs=jobs,
            results=results,
            in_flight=in_flight,
            file_timeout_s=file_timeout_s,
            timeout=0,
            ctx=ctx,
            converters=converters,
            cache_dir=cache_dir,
            src=src,
            worker_mem_mb=worker_mem_mb,
            loader=loader,
            worker_to_job=worker_to_job,
            idle_workers=idle_workers,
        )

    # Dispatch eager: send all idle workers immediately
    for fp, rel_key in dispatch_order:
        while not idle_workers:
            _collect(
                w_outs=w_outs,
                w_ins=w_ins,
                w_procs=w_procs,
                jobs=jobs,
                results=results,
                in_flight=in_flight,
                file_timeout_s=file_timeout_s,
                timeout=0.5,
                ctx=ctx,
                converters=converters,
                cache_dir=cache_dir,
                src=src,
                worker_mem_mb=worker_mem_mb,
                loader=loader,
                worker_to_job=worker_to_job,
                idle_workers=idle_workers,
            )

        w_idx = idle_workers.pop(0)
        try:
            w_ins[w_idx].send((str(fp), rel_key))
        except OSError:
            results[rel_key] = _make_error(rel_key, "unreadable", "parser crashed (send failed)")
        else:
            jobs[rel_key] = time.monotonic()
            in_flight[0] += 1
            worker_to_job[w_idx] = rel_key
        if window_check is not None:
            window_check(in_flight[0])

    # Phase 2: drain — use the same _collect with blocking wait
    while jobs:
        _collect(
            w_outs=w_outs,
            w_ins=w_ins,
            w_procs=w_procs,
            jobs=jobs,
            results=results,
            in_flight=in_flight,
            file_timeout_s=file_timeout_s,
            timeout=0.5,
            ctx=ctx,
            converters=converters,
            cache_dir=cache_dir,
            src=src,
            worker_mem_mb=worker_mem_mb,
            loader=loader,
            worker_to_job=worker_to_job,
            idle_workers=idle_workers,
        )

    # Final collection pass (non-blocking)
    _collect(
        w_outs=w_outs,
        w_ins=w_ins,
        w_procs=w_procs,
        jobs=jobs,
        results=results,
        in_flight=in_flight,
        file_timeout_s=file_timeout_s,
        timeout=0,
        ctx=ctx,
        converters=converters,
        cache_dir=cache_dir,
        src=src,
        worker_mem_mb=worker_mem_mb,
        loader=loader,
        worker_to_job=worker_to_job,
        idle_workers=idle_workers,
    )

    # Close pipes and reap workers
    for ins, outs, proc in zip(w_ins, w_outs, w_procs, strict=True):
        if ins is not None:
            ins.close()
        if outs is not None:
            outs.close()
        if proc is not None:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)

    # Safety net: any unresolved file is a bug
    if jobs:
        raise RuntimeError(
            f"supervisor_parse: {len(jobs)} jobs still unresolved: " + ", ".join(sorted(jobs))
        )

    return [(rk, results[rk]) for rk in sorted(results)]
