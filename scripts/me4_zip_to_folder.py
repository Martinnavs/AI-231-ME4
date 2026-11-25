"""Big-zip runbook: plan (dry run) or execute extract+ingest of a corpus zip (ticket 224)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from voice_agent.rag.zipingest import (
    MAX_MEMBER_BYTES,
    MAX_TOTAL_BYTES,
    _copy_bounded,
    validate_members,
)


def _free_disk(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _ingest_cmd(outdir: Path, workers: str, deadline_min: int, db: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "voice_agent.rag.ingest",
        str(outdir),
        "--workers",
        workers,
        "--deadline-min",
        str(deadline_min),
        "--out",
        db,
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="me4_zip_to_folder", description=__doc__)
    ap.add_argument("zip")
    ap.add_argument("outdir")
    ap.add_argument("--ingest", action="store_true", help="run the ingest CLI after extraction")
    ap.add_argument("--deadline-min", type=int, default=50)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--db", default="", help="ingest --out path (default: <outdir>.db)")
    ap.add_argument(
        "--execute", action="store_true", help="extract (and ingest) instead of printing the plan"
    )
    args = ap.parse_args(argv)
    zip_path, outdir = Path(args.zip), Path(args.outdir)
    if not zip_path.is_file():
        print(f"error: no such zip: {zip_path}", file=sys.stderr)
        return 2
    db = args.db or f"{outdir}.db"
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total_bytes = sum(i.file_size for i in infos)
    zip_size = zip_path.stat().st_size
    free = _free_disk(outdir)
    print(
        f"zip: {zip_path}  members: {len(infos)}  total_bytes: {total_bytes}  zip_bytes: {zip_size}"
    )
    print(f"free_disk: {free}  free >= 3x zip: {free >= 3 * zip_size}")
    cmd = _ingest_cmd(outdir, args.workers, args.deadline_min, db)
    if args.ingest:
        print("ingest: " + " ".join(cmd))
    if not args.execute:
        print("dry run: nothing extracted, nothing executed")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        accepted, refused, junk = validate_members(zf.infolist(), outdir)
        total = 0
        extracted = 0
        for info, target in accepted:
            limit = min(MAX_MEMBER_BYTES, MAX_TOTAL_BYTES - total)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    total += _copy_bounded(src, dst, limit)
                    extracted += 1
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                target.unlink(missing_ok=True)
                refused.append(f"{info.filename}: not extracted ({type(exc).__name__})")
    print(f"extracted: {extracted}  junk_skipped: {junk}")
    for r in refused:
        print(f"refused: {r}")
    if not args.ingest:
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if lines:
        print(lines[-1])
    if proc.returncode:
        print((proc.stderr or "")[-2000:], file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
