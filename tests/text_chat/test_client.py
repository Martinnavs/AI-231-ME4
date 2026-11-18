"""The text client: frames drawn on a UI, the history it sends, and a real WebSocket round trip."""

import contextlib
import json
import threading
from typing import Any

import pytest
from websockets.sync.server import ServerConnection, serve

from text_chat.cli import history_for, load_token
from text_chat.client import AgentClient, NoticeListener, render_turn


class FakeUi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args[0] if args else None))

        return record

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def streamed(self) -> str:
        return "".join(str(arg) for name, arg in self.calls if name == "stream_token")


def test_sentences_stream_with_a_space_between_and_the_spinner_stops_first() -> None:
    ui = FakeUi()
    result = render_turn(
        [{"text": "Jensen Huang is the CEO."}, {"text": "Source: Nvidia.", "flush": True},
         {"done": True}],
        ui,
    )  # fmt: skip
    assert result.text == "Jensen Huang is the CEO. Source: Nvidia."
    assert ui.streamed() == "Jensen Huang is the CEO. Source: Nvidia."
    assert ui.names() == ["turn_start", "start_thinking", "stop_thinking", "stream_start",
                          "stream_token", "stream_token", "stream_end"]  # fmt: skip


def test_a_tool_call_closes_the_text_and_shows_the_tool_until_the_next_text() -> None:
    ui = FakeUi()
    result = render_turn(
        [{"text": "Checking."}, {"event": "tool", "name": "weather", "said": True},
         {"text": "It is sunny."}, {"done": True}],
        ui,
    )  # fmt: skip
    assert result.text == "Checking. It is sunny."
    names = ui.names()
    assert names.count("stream_start") == 2 and names.count("stream_end") == 2
    assert ("show_tool_start", "weather") in ui.calls and ("show_tool_done", "weather") in ui.calls
    assert names.index("stop_tool_spinner") < len(names) - names[::-1].index("stream_start")


def test_other_events_and_errors_are_shown_not_dropped() -> None:
    ui = FakeUi()
    result = render_turn([{"event": "open_url", "url": "http://x/"}, {"error": "timeout"}], ui)
    assert result.error == "timeout" and result.text == ""
    notices = [arg for name, arg in ui.calls if name == "notice"]
    assert "open_url" in notices[0] and "http://x/" in notices[0] and "timeout" in notices[1]


def test_a_turn_with_no_text_ends_cleanly() -> None:
    ui = FakeUi()
    assert render_turn([{"done": True}], ui).text == ""
    assert "stream_start" not in ui.names() and ui.names().count("stop_thinking") == 1


def test_the_history_is_the_last_exchanges_and_the_new_message() -> None:
    old = [{"role": r, "content": str(i)} for i, r in enumerate(["user", "assistant"] * 5)]
    sent = history_for(old, "now", 2)
    assert [m["content"] for m in sent] == ["6", "7", "8", "9", "now"] and sent[-1][
        "role"
    ] == "user"
    assert history_for(old, "now", 0) == [{"role": "user", "content": "now"}]
    assert history_for([], "hi", 4) == [{"role": "user", "content": "hi"}]


def test_the_token_comes_from_the_environment_then_the_file(
    tmp_path: Any, monkeypatch: Any
) -> None:
    file = tmp_path / "token"
    file.write_text("from-file\n")
    monkeypatch.delenv("LAN_SPEECH_TOKEN", raising=False)
    assert load_token(file) == "from-file"
    monkeypatch.setenv("LAN_SPEECH_TOKEN", "from-env")
    assert load_token(file) == "from-env"
    assert load_token(tmp_path / "missing") == "from-env"
    monkeypatch.delenv("LAN_SPEECH_TOKEN")
    assert load_token(tmp_path / "missing") is None


@pytest.fixture
def server() -> Any:
    """A WebSocket server that speaks the agent server's protocol; remembers what it was sent."""
    seen: dict[str, Any] = {"requests": [], "auth": [], "acks": []}

    def handler(ws: ServerConnection) -> None:
        seen["auth"].append(ws.request.headers.get("Authorization"))
        request = json.loads(ws.recv())
        if request.get("notices"):
            ws.send(json.dumps({"notice": {"id": 7, "text": "Your page is ready.", "job": "j1"}}))
            with contextlib.suppress(Exception):  # the client may go away first
                seen["acks"].append(json.loads(ws.recv())["ack"])
            return
        seen["requests"].append(request)
        ws.send(json.dumps({"text": "Hello there."}))
        ws.send(json.dumps({"done": True}))

    with serve(handler, "127.0.0.1", 0) as srv:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        seen["url"] = f"ws://127.0.0.1:{srv.socket.getsockname()[1]}"
        yield seen
        srv.shutdown()


def test_a_turn_over_a_real_websocket_sends_the_messages_and_the_token(server: Any) -> None:
    ui = FakeUi()
    messages = [{"role": "user", "content": "hi"}]
    client = AgentClient(server["url"], "sekrit")
    result = render_turn(client.frames(messages), ui)
    assert result.text == "Hello there."
    assert server["requests"] == [{"messages": messages, "session": client.session}]
    assert server["auth"] == ["Bearer sekrit"]


def test_the_session_id_is_one_string_per_client_and_differs_between_clients(server: Any) -> None:
    first, second = AgentClient(server["url"], None), AgentClient(server["url"], None)
    assert isinstance(first.session, str) and 0 < len(first.session) <= 64
    assert first.session != second.session
    for client in (first, first):
        render_turn(client.frames([{"role": "user", "content": "hi"}]), FakeUi())
    assert {r["session"] for r in server["requests"]} == {first.session}


def test_a_notice_arrives_and_the_ack_goes_back(server: Any) -> None:
    listener = NoticeListener(server["url"], None)
    listener.start()
    note = listener.notices.get(timeout=5)
    assert note["text"] == "Your page is ready."
    listener.ack(note)
    for _ in range(50):
        if server["acks"]:
            break
        threading.Event().wait(0.1)
    assert server["acks"] == [7] and server["auth"][0] is None
