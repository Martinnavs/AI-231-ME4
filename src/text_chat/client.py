"""The agent server's WebSocket protocol, drawn on a `ChatUI`.

A turn sends `{"messages": [...]}` and reads frames until `{"done": true}` or `{"error": ...}`:
`{"text": "..."}` is a sentence (the voice client speaks it), `{"event": "tool", "name": ...}` a
tool call starting, any other `{"event": ...}` something the voice client acts on (open a page).
Finished background jobs arrive on a second connection that starts with `{"notices": true}`, each
as `{"notice": {"id", "text", "job"}}` and again on every reconnect until it is acked.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect


class Ui(Protocol):
    """The `ChatUI` calls a turn makes."""

    def turn_start(self) -> None: ...
    def start_thinking(self) -> None: ...
    def stop_thinking(self) -> None: ...
    def stream_start(self) -> None: ...
    def stream_token(self, token: str) -> None: ...
    def stream_end(self, elapsed: str = "") -> None: ...
    def show_tool_start(self, name: str, arguments: dict[str, Any]) -> None: ...
    def start_tool_spinner(self, name: str, arguments: dict[str, Any]) -> None: ...
    def stop_tool_spinner(self) -> None: ...
    def show_tool_done(self, name: str, result: str, success: bool = True) -> None: ...
    def notice(
        self, message: str, level: Literal["info", "warning", "error", "debug"] = "info"
    ) -> None: ...


@dataclass
class TurnResult:
    text: str = ""  # what the agent said, for the history the next turn sends
    error: str = ""


def render_turn(frames: Iterable[dict[str, Any]], ui: Ui) -> TurnResult:
    """Draw one turn's frames; returns what was said."""
    ui.turn_start()
    ui.start_thinking()
    thinking, streaming, tool = True, False, ""
    spoken: list[str] = []
    result = TurnResult()

    def settle() -> None:
        nonlocal thinking, tool
        if thinking:
            ui.stop_thinking()
            thinking = False
        if tool:
            ui.stop_tool_spinner()
            ui.show_tool_done(tool, "")
            tool = ""

    def close_stream() -> None:
        nonlocal streaming
        if streaming:
            ui.stream_end()
            streaming = False

    for frame in frames:
        if "text" in frame and frame["text"]:
            settle()
            if not streaming:
                ui.stream_start()
                streaming = True
            piece = str(frame["text"])
            if spoken and not spoken[-1].endswith((" ", "\n")) and not piece.startswith(" "):
                piece = " " + piece  # the server cuts at sentences: the space went with the cut
            spoken.append(piece)
            ui.stream_token(piece)
        elif frame.get("event") == "tool":
            settle()
            close_stream()
            tool = str(frame.get("name", "tool"))
            ui.show_tool_start(tool, {})
            ui.start_tool_spinner(tool, {})
        elif "event" in frame:
            detail = {k: v for k, v in frame.items() if k != "event"}
            ui.notice(f"{frame['event']}: {json.dumps(detail, ensure_ascii=False)}")
        elif "error" in frame:
            result.error = str(frame["error"])
            break
        elif frame.get("done"):
            break
    settle()
    close_stream()
    if result.error:
        ui.notice(f"The agent server answered: {result.error}", level="error")
    result.text = "".join(spoken).strip()
    return result


def headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


class AgentClient:
    def __init__(self, url: str, token: str | None) -> None:
        self.url, self.token = url, token

    def frames(self, messages: list[dict[str, str]]) -> Iterator[dict[str, Any]]:
        """One turn's frames. Connection errors are raised for the caller to show."""
        with connect(
            self.url, additional_headers=headers(self.token), open_timeout=10, max_size=None
        ) as ws:
            ws.send(json.dumps({"messages": messages}))
            for raw in ws:
                frame = json.loads(raw)
                yield frame
                if "done" in frame or "error" in frame:
                    return


class NoticeListener(threading.Thread):
    """Collects the finished-job notices in a queue; `ack` tells the server one was shown."""

    def __init__(self, url: str, token: str | None) -> None:
        super().__init__(daemon=True)
        self.url, self.token = url, token
        self.notices: queue.Queue[dict[str, Any]] = queue.Queue()
        self.ws: ClientConnection | None = None
        self.seen: set[int] = set()

    def run(self) -> None:
        while True:
            try:
                with connect(
                    self.url, additional_headers=headers(self.token), open_timeout=10
                ) as ws:
                    self.ws = ws
                    ws.send(json.dumps({"notices": True}))
                    for raw in ws:
                        note = json.loads(raw).get("notice")
                        if isinstance(note, dict) and note.get("id") not in self.seen:
                            self.seen.add(note["id"])
                            self.notices.put(note)
            except (OSError, WebSocketException, ValueError):
                pass
            self.ws = None
            time.sleep(3)  # the server is down or restarting: try again

    def ack(self, note: dict[str, Any]) -> None:
        with contextlib.suppress(OSError, WebSocketException, AttributeError):
            if self.ws is not None:
                self.ws.send(json.dumps({"ack": note["id"]}))
