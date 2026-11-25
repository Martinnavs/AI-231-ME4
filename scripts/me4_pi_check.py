"""One-shot measurement for the Pi: ingest a zip/folder, check the index, time queries and OCR.

Run on the Pi from the repo root (no sudo, stops no services):
    PYTHONPATH=src uv run python scripts/me4_pi_check.py SRC --work /tmp/pi-check --out pi.json
Everything it prints is measured on the machine it runs on. Paste pi.json back.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_QUERIES = [
    "What is this document about?",
    "How many pages or rows does the largest file have?",
    "Which file mentions a deadline or a date?",
    "What is the total or the maximum value in the spreadsheet?",
    "Who is the author or the contact person?",
]


def _facts(work: Path) -> dict[str, Any]:
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            mem[key] = int(rest.split()[0]) // 1024
    cpu = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in Path("/proc/cpuinfo").read_text().splitlines()
            if ln.startswith(("Model name", "model name", "Model"))
        ),
        "?",
    )
    throttled = ""
    if shutil.which("vcgencmd"):
        throttled = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, check=False
        ).stdout.strip()
    return {
        "machine": platform.machine(),
        "cpu": cpu,
        "mem_mb": mem,
        "disk_free_mb": shutil.disk_usage(work).free // (1024 * 1024),
        "throttled": throttled,
        "python": sys.version.split()[0],
    }


def _ingest(src: Path, db: Path) -> dict[str, Any]:
    for p in (db, Path(str(db) + ".fts")):
        p.unlink(missing_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "voice_agent.rag.ingest", str(src), "--out", str(db)],
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    return {
        "exit": proc.returncode,
        "wall_s": round(wall, 1),
        "parent_peak_mb": peak_mb,
        "last_line": (proc.stdout.strip().splitlines() or [""])[-1],
        "stderr_tail": proc.stderr.strip()[-400:],
    }


def _integrity(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db))
    fts = sqlite3.connect(str(db) + ".fts")
    try:
        files = dict(conn.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall())
        chunks = conn.execute("SELECT count(*) FROM chunk_data").fetchone()[0]
        indexed = fts.execute("SELECT count(*) FROM chunks").fetchone()[0]
        images = dict(
            conn.execute("SELECT status, count(*) FROM images GROUP BY status").fetchall()
        )
    finally:
        conn.close()
        fts.close()
    return {
        "files": files,
        "chunks": chunks,
        "search_index_rows": indexed,
        "index_matches_chunks": chunks == indexed,  # the bug that cost us a day
        "images": images,
    }


def _queries(db: Path, raw: Path, questions: list[str]) -> dict[str, Any]:
    from voice_agent.rag.dbindex import DbIndex

    idx = DbIndex(db, raw_root=raw)
    times, out = [], []
    try:
        for q in questions:
            t0 = time.perf_counter()
            ctx = idx.file_context(q)
            dt = time.perf_counter() - t0
            times.append(dt)
            out.append({"q": q, "s": round(dt, 2), "files": sorted({p["source"] for p in ctx})[:3]})
    finally:
        idx.close()
    return {
        "median_s": round(statistics.median(times), 2),
        "max_s": round(max(times), 2),
        "rows": out,
    }


def _ocr_bench(db: Path, raw: Path, n: int) -> dict[str, Any]:
    from voice_agent.rag import ocr
    from voice_agent.rag.images import ImageRef

    if not ocr.engine_available():
        return {"engine": None, "note": "no rapidocr models and no tesseract binary"}
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT rel_key, locator, kind, w, h, sha256 FROM images GROUP BY sha256"
            " ORDER BY CASE kind WHEN 'pdf_scan_page' THEN 0 ELSE 1 END LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        conn.close()
    session, secs, chars = ocr.OcrSession(), [], []
    for rel_key, locator, kind, w, h, sha in rows:
        ref = ImageRef(rel_key=rel_key, locator=locator, kind=kind, w=w, h=h, nbytes=0, sha256=sha)
        try:
            raw_bytes = ocr.extract_image_bytes(ref, raw)
        except OSError:
            continue
        t0 = time.perf_counter()
        text, _conf, _note = ocr.ocr_image(
            ref, raw_bytes, session, max_long=1280 if kind == "pdf_scan_page" else 960
        )
        secs.append(time.perf_counter() - t0)
        chars.append(len(text))
    hwm = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            hwm = int(line.split()[1]) // 1024
    if not secs:
        return {"engine": "available", "images": 0}
    secs.sort()
    return {
        "engine": "available",
        "images": len(secs),
        "total_s": round(sum(secs), 1),
        "median_s": round(statistics.median(secs), 2),
        "p90_s": round(secs[int(0.9 * (len(secs) - 1))], 2),
        "images_with_text": sum(1 for c in chars if c > 0),
        "peak_rss_mb": hwm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="zip or folder to ingest")
    ap.add_argument("--work", default="/tmp/pi-check")
    ap.add_argument("--out", default="pi-check.json")
    ap.add_argument("--queries", help="text file, one question per line")
    ap.add_argument("--ocr-images", type=int, default=30)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)
    db = work / "docs.db"
    report: dict[str, Any] = {"facts": _facts(work)}
    report["ingest"] = _ingest(src, db)
    print(json.dumps(report["ingest"]), flush=True)
    if report["ingest"]["exit"] == 0:
        report["integrity"] = _integrity(db)
        print(json.dumps(report["integrity"]), flush=True)
        raw = src if src.is_dir() else work / "raw"
        if src.is_file() and not raw.exists():
            shutil.unpack_archive(str(src), str(raw))
        qs = (
            [ln for ln in Path(args.queries).read_text().splitlines() if ln.strip()]
            if args.queries
            else DEFAULT_QUERIES
        )
        report["queries"] = _queries(db, raw, qs)
        report["ocr"] = _ocr_bench(db, raw, args.ocr_images)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0 if report["ingest"]["exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
