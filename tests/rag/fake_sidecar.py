"""Pure-Python fake sidecar script for testing ``SidecarClient`` failure modes.

Imports only the Python standard library (never ``fastembed`` or ``voice_agent``).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


def _load_behavior(cache_dir: Path, default_mode: str, op: str) -> str:
    cfg_path = cache_dir / "fake_behavior.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if op in data and isinstance(data[op], str):
                    return data[op]
                if "default" in data and isinstance(data["default"], str):
                    return data["default"]
        except Exception:
            pass
    return default_mode


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fake embedder sidecar for tests")
    parser.add_argument("--cache-dir", type=Path, default=Path("."))
    parser.add_argument("--model", default="fake/model")
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", default="ok")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    mode = args.backend
    if args.check:
        if mode == "fail_check":
            _write_json({"ok": False, "error": "fake check failed"})
            return 2
        if mode == "hang_check":
            time.sleep(5.0)
            return 2
        _write_json({"ok": True, "model": args.model, "dim": 16})
        return 0

    if mode == "hang_ignore_sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req = json.loads(line)
        req_id = int(req["id"])
        op = str(req["op"])

        eff_mode = _load_behavior(args.cache_dir, mode, op)

        if op == "ping" and eff_mode != "hang_ping":
            _write_json(
                {
                    "id": req_id,
                    "ok": True,
                    "model": args.model,
                    "dim": 16,
                    "backend": mode,
                    "pid": os.getpid(),
                }
            )
            continue

        if eff_mode in ("hang", "hang_ping", "hang_ignore_sigterm"):
            time.sleep(30.0)
            return 0
        if eff_mode == "eof":
            return 1
        if eff_mode == "garbage":
            sys.stdout.write("not valid json {{{\n")
            sys.stdout.flush()
            continue
        if eff_mode == "wrong_id":
            _write_json({"id": req_id + 999, "ok": True})
            continue
        if eff_mode == "ok_false":
            _write_json({"id": req_id, "ok": False, "error": "fake failure"})
            continue
        if eff_mode == "large_5mb":
            pad = "x" * (5 * 1024 * 1024)
            _write_json({"id": req_id, "ok": True, "done": list(req.get("keys", [])), "pad": pad})
            continue

        if op == "embed_passages":
            _write_json({"id": req_id, "ok": True, "done": list(req.get("keys", []))})
        elif op == "score_query":
            keys = list(req.get("keys", []))
            ranking = [[idx, float(len(keys) - idx)] for idx in range(len(keys))]
            _write_json({"id": req_id, "ok": True, "ranking": ranking, "missing": []})
        else:
            _write_json({"id": req_id, "ok": False, "error": f"unknown_op: {op}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
