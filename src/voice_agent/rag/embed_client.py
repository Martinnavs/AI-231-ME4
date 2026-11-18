"""Subprocess stdio client for the RAG late-interaction embedder sidecar."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import selectors
import subprocess
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MAX_LINE_BYTES = 16 * 1024 * 1024
_CHECK_TIMEOUT_S = 60.0
_CLOSE_WAIT_S = 2.0
_TERMINATE_WAIT_S = 0.5


class SidecarError(Exception):
    """Raised on any embedder sidecar lifecycle, protocol, or remote error."""


def _reduced_env() -> dict[str, str]:
    env: dict[str, str] = {"HF_HUB_OFFLINE": "1"}
    for key in ("PATH", "HOME"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def _close_on_exit(client_ref: weakref.ref[SidecarClient]) -> None:
    client = client_ref()
    if client is not None:
        with contextlib.suppress(Exception):
            client.close()


class SidecarClient:
    """Thread-safe stdio JSON-lines client for ``embedder_sidecar.py``."""

    def __init__(
        self,
        python: str | Path,
        script: Path | str,
        cache_dir: Path | str,
        model: str,
        model_dir: Path | str,
        threads: int = 4,
        backend: str = "fastembed",
        start_timeout: float = 120.0,
        request_timeout: float = 30.0,
    ) -> None:
        self._python = str(python)
        self._script = Path(script)
        self._cache_dir = Path(cache_dir)
        self._model = model
        self._model_dir = Path(model_dir)
        self._threads = threads
        self._backend = backend
        self._start_timeout = float(start_timeout)
        self._request_timeout = float(request_timeout)

        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._rx_buf = bytearray()
        self._next_id = 1
        atexit.register(_close_on_exit, weakref.ref(self))

    @property
    def alive(self) -> bool:
        """True when the sidecar child process is running."""
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _base_cmd(self) -> list[str]:
        return [
            self._python,
            "-I",
            str(self._script),
            "--cache-dir",
            str(self._cache_dir),
            "--model",
            self._model,
            "--model-dir",
            str(self._model_dir),
            "--threads",
            str(self._threads),
            "--backend",
            self._backend,
        ]

    def check(self) -> str | None:
        """Run ``--check`` and return ``None`` if ready, else a failure reason."""
        cmd = [
            self._python,
            "-I",
            str(self._script),
            "--check",
            "--model",
            self._model,
            "--model-dir",
            str(self._model_dir),
            "--threads",
            str(self._threads),
            "--backend",
            self._backend,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_CHECK_TIMEOUT_S,
                env=_reduced_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"sidecar --check timed out after {_CHECK_TIMEOUT_S:.0f}s"
        except OSError as exc:
            return f"failed to run sidecar --check: {exc}"
        except Exception as exc:
            return f"sidecar --check failed: {exc}"

        raw_out = (proc.stdout or "").strip()
        last_line = raw_out.splitlines()[-1] if raw_out else ""
        if proc.returncode == 0 and last_line:
            try:
                parsed = json.loads(last_line)
                if isinstance(parsed, dict) and parsed.get("ok") is True:
                    return None
                if isinstance(parsed, dict) and parsed.get("error"):
                    return str(parsed["error"])
            except json.JSONDecodeError:
                pass
            return f"sidecar --check returned invalid JSON: {last_line!r}"

        if last_line:
            try:
                parsed = json.loads(last_line)
                if isinstance(parsed, dict) and parsed.get("error"):
                    return str(parsed["error"])
            except json.JSONDecodeError:
                pass
            return last_line

        err_out = (proc.stderr or "").strip()
        if err_out:
            return err_out.splitlines()[-1]
        return f"sidecar --check exited with status {proc.returncode}"

    def start(self) -> None:
        """Spawn the sidecar child process and verify readiness with ``ping``."""
        with self._lock:
            if self._proc is not None:
                if self._proc.poll() is None:
                    return
                self._kill_locked()

            try:
                proc = subprocess.Popen(
                    self._base_cmd(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=_reduced_env(),
                )
            except OSError as exc:
                raise SidecarError(f"failed to start sidecar: {exc}") from exc
            except Exception as exc:
                raise SidecarError(f"failed to start sidecar: {exc}") from exc

            if proc.stdin is None or proc.stdout is None:
                self._proc = proc
                self._kill_locked()
                raise SidecarError("sidecar started without stdin/stdout pipes")

            try:
                os.set_blocking(proc.stdout.fileno(), False)
            except OSError as exc:
                self._proc = proc
                self._kill_locked()
                raise SidecarError(f"failed to configure sidecar stdout pipe: {exc}") from exc

            self._proc = proc
            self._rx_buf.clear()

            try:
                self._request_locked("ping", None, timeout=self._start_timeout)
            except SidecarError:
                self._kill_locked()
                raise

    def request(
        self,
        op: str,
        payload: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one JSON-lines request and wait for its matching response."""
        with self._lock:
            return self._request_locked(op, payload, timeout=timeout)

    def _request_locked(
        self,
        op: str,
        payload: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            if proc is not None:
                self._kill_locked()
            raise SidecarError("sidecar process is not running")
        if proc.stdin is None or proc.stdout is None:
            self._kill_locked()
            raise SidecarError("sidecar stdio pipes are unavailable")

        eff_timeout = self._request_timeout if timeout is None else float(timeout)
        if eff_timeout <= 0.0:
            self._kill_locked()
            raise SidecarError(f"invalid request timeout: {eff_timeout}")

        req_id = self._next_id
        self._next_id += 1
        req: dict[str, Any] = dict(payload) if payload is not None else {}
        req["id"] = req_id
        req["op"] = op

        try:
            wire = (json.dumps(req, allow_nan=False) + "\n").encode("utf-8")
            proc.stdin.write(wire)
            proc.stdin.flush()
        except (OSError, ValueError, TypeError) as exc:
            self._kill_locked()
            raise SidecarError(f"failed writing {op!r} request to sidecar: {exc}") from exc

        raw_line = self._read_line_with_deadline_locked(proc, op, eff_timeout)

        try:
            resp = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._kill_locked()
            raise SidecarError(f"sidecar returned malformed JSON for {op!r}: {exc}") from exc

        if not isinstance(resp, dict):
            self._kill_locked()
            raise SidecarError(f"sidecar returned non-object JSON for {op!r}")

        resp_id = resp.get("id")
        if type(resp_id) is not int or resp_id != req_id:
            self._kill_locked()
            raise SidecarError(
                f"sidecar response id mismatch for {op!r}: expected {req_id}, got {resp_id!r}"
            )

        if resp.get("ok") is not True:
            err_msg = resp.get("error")
            raise SidecarError(str(err_msg) if err_msg else f"sidecar {op!r} failed")

        return resp

    def _read_line_with_deadline_locked(
        self,
        proc: subprocess.Popen[bytes],
        op: str,
        timeout: float,
    ) -> bytes:
        assert proc.stdout is not None
        stdout_fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout

        with selectors.DefaultSelector() as sel:
            sel.register(stdout_fd, selectors.EVENT_READ)
            while True:
                newline_idx = self._rx_buf.find(b"\n")
                if newline_idx != -1:
                    line = bytes(self._rx_buf[:newline_idx])
                    del self._rx_buf[: newline_idx + 1]
                    return line

                if len(self._rx_buf) > MAX_LINE_BYTES:
                    self._kill_locked()
                    raise SidecarError(
                        f"sidecar response exceeded {MAX_LINE_BYTES} bytes during {op!r}"
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._kill_locked()
                    raise SidecarError(f"sidecar request {op!r} timed out after {timeout:.2f}s")

                try:
                    events = sel.select(timeout=remaining)
                except OSError as exc:
                    self._kill_locked()
                    raise SidecarError(f"selector error waiting for {op!r}: {exc}") from exc

                if not events:
                    self._kill_locked()
                    raise SidecarError(f"sidecar request {op!r} timed out after {timeout:.2f}s")

                while True:
                    try:
                        chunk = os.read(stdout_fd, 65536)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        self._kill_locked()
                        raise SidecarError(f"sidecar read error during {op!r}: {exc}") from exc

                    if not chunk:
                        self._kill_locked()
                        raise SidecarError(f"sidecar closed stdout (EOF) during {op!r}")

                    self._rx_buf.extend(chunk)
                    if len(self._rx_buf) > MAX_LINE_BYTES and b"\n" not in self._rx_buf:
                        self._kill_locked()
                        raise SidecarError(
                            f"sidecar response exceeded {MAX_LINE_BYTES} bytes during {op!r}"
                        )
                    if b"\n" in chunk:
                        break

    def _kill_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._rx_buf.clear()
        if proc is None:
            return

        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()

        if proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=_TERMINATE_WAIT_S)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=_CLOSE_WAIT_S)

    def ping(self) -> dict[str, Any]:
        """Ping the running sidecar and return its status dictionary."""
        return self.request("ping")

    def embed_passages(self, keys: Sequence[str], texts: Sequence[str]) -> list[str]:
        """Embed ``texts`` under ``keys`` into the on-disk cache."""
        resp = self.request(
            "embed_passages",
            {"keys": list(keys), "texts": list(texts)},
        )
        done = resp.get("done")
        if not isinstance(done, list) or not all(isinstance(k, str) for k in done):
            with self._lock:
                self._kill_locked()
            raise SidecarError("sidecar embed_passages returned invalid 'done' field")
        return list(done)

    def score_query(self, query: str, keys: Sequence[str]) -> list[tuple[int, float]]:
        """Score ``query`` against all ``keys`` and return ``[(position, score), ...]``."""
        resp = self.request(
            "score_query",
            {"query": query, "keys": list(keys)},
        )
        ranking = resp.get("ranking")
        if not isinstance(ranking, list):
            with self._lock:
                self._kill_locked()
            raise SidecarError("sidecar score_query returned invalid 'ranking' field")
        parsed: list[tuple[int, float]] = []
        for item in ranking:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or type(item[0]) is not int
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, float))
            ):
                with self._lock:
                    self._kill_locked()
                raise SidecarError(
                    f"sidecar score_query returned malformed ranking entry: {item!r}"
                )
            parsed.append((item[0], float(item[1])))
        return parsed

    def close(self) -> None:
        """Idempotently shut down the sidecar via stdin EOF, then kill if needed."""
        with self._lock:
            proc = self._proc
            self._proc = None
            self._rx_buf.clear()
            if proc is None:
                return

            if proc.stdin is not None:
                with contextlib.suppress(OSError):
                    proc.stdin.close()

            if proc.poll() is None:
                try:
                    proc.wait(timeout=_CLOSE_WAIT_S)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=_CLOSE_WAIT_S)

            if proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()
