"""Real-audio timing and turn-join telemetry tests for doc_chat (ticket 265)."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import socket
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("websockets")
pytest.importorskip("langchain_core")

from websockets.sync.server import ServerConnection, serve  # noqa: E402

import doc_chat.playback as playback_mod  # noqa: E402
from doc_chat import fdbench_marks  # noqa: E402
from doc_chat.__main__ import Session  # noqa: E402
from doc_chat.agent_client import AgentChatModel  # noqa: E402
from doc_chat.chain import build_sentence_streamer  # noqa: E402
from doc_chat.dialogue import ChatHistory  # noqa: E402
from doc_chat.fillers import Fillers  # noqa: E402
from doc_chat.speech import HeardTurn, SpokenTurn  # noqa: E402

FILLER_CONFIG = {
    "timing": {"stall_s": 0.02, "notice_s": 0.5, "giveup_s": 2.0},
    "stall": ["One moment."],
    "notice": ["Still working on that."],
    "giveup": ["Sorry, that took too long."],
    "default": ["Checking."],
    "tools": {"web_answer": ["Let me look that up."]},
}


class _FakePlayer:
    """In-memory player recording pushed chunks and delaying close() slightly."""

    def __init__(self, close_delay_s: float = 0.03) -> None:
        self.pushed: list[bytes] = []
        self.closed = False
        self.underruns = 0
        self.close_delay_s = close_delay_s

    def push(self, pcm: bytes) -> None:
        self.pushed.append(pcm)

    def close(self) -> None:
        if self.close_delay_s > 0:
            time.sleep(self.close_delay_s)
        self.closed = True

    def abort(self) -> None:
        self.closed = True


class _FakeSpeechClient:
    """Fake speech client driving on_chunk/on_sentence without network or audio devices."""

    tts_sample_rate = 24000
    noise_floor: float | None = None
    hub = None
    canceller = None

    def __init__(
        self,
        *,
        chunk_delay_s: float = 0.0,
        emit_audio: bool = True,
        heard_queue: list[HeardTurn] | None = None,
    ) -> None:
        self.chunk_delay_s = chunk_delay_s
        self.emit_audio = emit_audio
        self.heard_queue = list(heard_queue or [])
        self.spoken_sentences: list[str] = []
        self.chunk_times: list[tuple[str, float]] = []

    async def hear_wav(
        self, wav_path: Path | str, chunk_ms: int = 100, **_: Any
    ) -> HeardTurn:
        del wav_path, chunk_ms
        return self.heard_queue.pop(0)

    async def speak_text(
        self,
        text: str,
        voice: str | None = "F1",
        t_speech_end: float | None = None,
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> tuple[bytes, float, float, float]:
        del voice
        self.spoken_sentences.append(text)
        t0 = time.perf_counter()
        if self.chunk_delay_s > 0:
            await asyncio.sleep(self.chunk_delay_s)
        if not self.emit_audio:
            elapsed = (time.perf_counter() - t0) * 1000.0
            to_first = elapsed if t_speech_end is None else (t0 - t_speech_end) * 1000.0
            return b"", elapsed, elapsed, to_first
        pcm = b"\x01\x00" * 240
        t_chunk = time.perf_counter()
        self.chunk_times.append((text, t_chunk))
        if on_chunk is not None:
            on_chunk(pcm)
        first_ms = (t_chunk - t0) * 1000.0
        to_first = first_ms if t_speech_end is None else (t_chunk - t_speech_end) * 1000.0
        return pcm, first_ms, first_ms, to_first

    async def speak_sentences(
        self,
        sentences: Iterable[str],
        voice: str | None = "F1",
        t_speech_end: float | None = None,
        on_chunk: Callable[[bytes], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
    ) -> SpokenTurn:
        iterator = iter(sentences)
        done = object()
        loop = asyncio.get_running_loop()
        parts: list[bytes] = []
        spoken: list[str] = []
        first_audio_ms = 0.0
        speech_end_to_first = 0.0
        t0 = time.perf_counter()
        while True:
            sentence = await loop.run_in_executor(None, lambda: next(iterator, done))
            if sentence is done:
                break
            assert isinstance(sentence, str)
            if on_sentence is not None:
                on_sentence(sentence)
            pcm, first_ms, _total, to_first = await self.speak_text(
                sentence,
                voice=voice,
                t_speech_end=t_speech_end if not spoken else None,
                on_chunk=on_chunk,
            )
            if not spoken:
                first_audio_ms, speech_end_to_first = first_ms, to_first
            parts.append(pcm)
            spoken.append(sentence)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return SpokenTurn(b"".join(parts), first_audio_ms, total_ms, speech_end_to_first, spoken)

    async def send_tts_cancel(self) -> None:
        return None


class _ScriptedWsServer:
    """Local loopback WebSocket server replaying scripted frames with optional delays."""

    def __init__(self, steps: list[list[tuple[float, dict[str, Any]]]]) -> None:
        self.steps = list(steps)
        self.received: list[dict[str, Any]] = []

    def handle(self, ws: ServerConnection) -> None:
        self.received.append(json.loads(ws.recv()))
        frames = self.steps.pop(0)
        for delay_s, frame in frames:
            if delay_s > 0:
                time.sleep(delay_s)
            ws.send(json.dumps(frame))


def _serve_frames(
    steps: list[list[tuple[float, dict[str, Any]]]],
) -> tuple[Any, _ScriptedWsServer, str]:
    srv = _ScriptedWsServer(steps)
    server = serve(srv.handle, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, srv, f"ws://127.0.0.1:{port}"


def _make_session(
    client: _FakeSpeechClient,
    model: AgentChatModel | None = None,
    *,
    play: bool = True,
    wav: list[str] | None = None,
    fillers: Fillers | None = None,
) -> Session:
    args = argparse.Namespace(
        play=play,
        play_remote=None,
        voice="F1",
        slot="doc?",
        risk=None,
        vad_hold=0.6,
        barge_in=False,
        llm="agent",
        wav=wav or [],
        addressee="off",
        voice_gate_db=0.0,
        cue_beep=0.0,
        cue_every=0,
    )
    streamer = build_sentence_streamer(None, model, max_words=60) if model is not None else None
    s = Session(
        args=args,
        client=client,  # type: ignore[arg-type]
        log_file=io.StringIO(),
        out_dir=None,
        streamer=streamer,
        history=ChatHistory(4),
        fillers=fillers,
    )
    if model is not None:
        model.on_event = s.on_agent_event
    return s


def _fillers() -> Fillers:
    f = Fillers(FILLER_CONFIG)
    f.audio = {line: b"\x02\x00" * 120 for line in f.lines()}
    return f


async def test_filler_then_real_answer_logs_real_audio_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = _FakePlayer(close_delay_s=0.03)
    monkeypatch.setattr(playback_mod, "make_player", lambda *a, **k: player)
    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.08, {"text": "The answer is forty two."}),
            (0.0, {"done": True, "turn": "run-0001"}),
        ]
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        t_end = time.perf_counter()
        heard = HeardTurn("What is the answer?", None, None, t_end, t_end, t_end)
        client = _FakeSpeechClient(heard_queue=[heard])
        model = AgentChatModel(url=url)
        session = _make_session(client, model, wav=["q1.wav"], fillers=_fillers())
        await session.run_wavs()
    finally:
        ws_srv.shutdown()

    rec = session.records[-1]
    assert rec["fillers"] == ["One moment."]
    assert rec["filler_log"] is not None
    assert rec["first_say_audio_ms"] is None
    assert rec["first_real_audio_ms"] is not None
    assert rec["first_real_audio_ms"] >= 75.0
    assert rec["full_answer_ms"] is not None
    assert rec["full_answer_ms"] >= rec["first_real_audio_ms"] + 20.0
    assert isinstance(rec["t_speech_end_wall"], float)
    assert rec["agent_turn_id"] == "run-0001"


async def test_say_line_then_real_answer_and_audio_out_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.2)
    monkeypatch.setenv("FDBENCH_MARKS", f"udp://127.0.0.1:{sock.getsockname()[1]}")
    monkeypatch.setattr(fdbench_marks, "_udp_checked", False)
    monkeypatch.setattr(fdbench_marks, "_udp", None)

    player = _FakePlayer(close_delay_s=0.02)
    monkeypatch.setattr(playback_mod, "make_player", lambda *a, **k: player)
    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.01, {"text": "Let me look that up.", "flush": True}),
            (0.06, {"text": "Lima is the capital of Peru."}),
            (0.0, {"done": True, "turn": "run-0002"}),
        ]
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        t_end = time.perf_counter()
        heard = HeardTurn("What is the capital of Peru?", None, None, t_end, t_end, t_end)
        client = _FakeSpeechClient(heard_queue=[heard])
        model = AgentChatModel(url=url)
        session = _make_session(client, model, wav=["q1.wav"])
        await session.run_wavs()

        marks: list[dict[str, Any]] = []
        try:
            while True:
                marks.append(json.loads(sock.recv(65536)))
        except TimeoutError:
            pass
    finally:
        ws_srv.shutdown()
        sock.close()
        monkeypatch.setattr(fdbench_marks, "_udp_checked", False)
        monkeypatch.setattr(fdbench_marks, "_udp", None)

    rec = session.records[-1]
    assert rec["first_say_audio_ms"] is not None
    assert rec["first_real_audio_ms"] is not None
    assert rec["first_real_audio_ms"] > rec["first_say_audio_ms"] + 40.0
    assert rec["full_answer_ms"] is not None
    assert rec["full_answer_ms"] >= rec["first_real_audio_ms"]
    assert rec["agent_turn_id"] == "run-0002"

    # audio_out must not fire on the `say` chunk; it fires once on the real answer chunk.
    audio_out_marks = [m for m in marks if m["stage"] == "audio_out"]
    assert len(audio_out_marks) == 1
    say_chunk_t = client.chunk_times[0][1]
    real_chunk_t = client.chunk_times[1][1]
    assert client.chunk_times[0][0] == "Let me look that up."
    assert client.chunk_times[1][0] == "Lima is the capital of Peru."
    assert real_chunk_t > say_chunk_t


async def test_prefix_then_real_answer_does_not_stop_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = _FakePlayer(close_delay_s=0.02)
    monkeypatch.setattr(playback_mod, "make_player", lambda *a, **k: player)
    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.0, {"text": "Your documents don't cover that. ", "meta": "prefix"}),
            (0.07, {"text": "The capital of Peru is Lima."}),
            (0.0, {"done": True, "turn": "run-0003"}),
        ]
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        t_end = time.perf_counter()
        heard = HeardTurn(
            "According to the memo, capital of Peru?", None, None, t_end, t_end, t_end
        )
        client = _FakeSpeechClient(heard_queue=[heard])
        model = AgentChatModel(url=url)
        session = _make_session(client, model, wav=["q1.wav"])
        await session.run_wavs()
    finally:
        ws_srv.shutdown()

    assert client.spoken_sentences == [
        "Your documents don't cover that.",
        "The capital of Peru is Lima.",
    ]
    rec = session.records[-1]
    assert rec["first_say_audio_ms"] is None
    assert rec["first_real_audio_ms"] is not None
    # Prefix ran before the 70 ms delay; real answer arrived after it.
    assert rec["first_real_audio_ms"] >= 60.0
    assert rec["full_answer_ms"] is not None
    assert rec["full_answer_ms"] >= rec["first_real_audio_ms"]
    assert rec["agent_turn_id"] == "run-0003"


async def test_no_real_answer_audio_produces_nulls_never_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = _FakePlayer(close_delay_s=0.01)
    monkeypatch.setattr(playback_mod, "make_player", lambda *a, **k: player)
    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.0, {"text": "Just a status line.", "flush": True}),
            (0.0, {"done": True, "turn": "run-0004"}),
        ],
        [
            (0.0, {"text": "Some answer."}),
            (0.0, {"done": True, "turn": "run-0005"}),
        ],
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        t_end = time.perf_counter()
        h1 = HeardTurn("Q1", None, None, t_end, t_end, t_end)
        client1 = _FakeSpeechClient(emit_audio=True, heard_queue=[h1])
        model1 = AgentChatModel(url=url)
        s1 = _make_session(client1, model1, wav=["q1.wav"])
        await s1.run_wavs()

        t_end2 = time.perf_counter()
        h2 = HeardTurn("Q2", None, None, t_end2, t_end2, t_end2)
        client2 = _FakeSpeechClient(emit_audio=False, heard_queue=[h2])
        model2 = AgentChatModel(url=url)
        s2 = _make_session(client2, model2, wav=["q2.wav"])
        await s2.run_wavs()
    finally:
        ws_srv.shutdown()

    rec1 = s1.records[-1]
    assert rec1["first_say_audio_ms"] is not None
    assert rec1["first_real_audio_ms"] is None
    assert rec1["full_answer_ms"] is None
    assert rec1["agent_turn_id"] == "run-0004"

    rec2 = s2.records[-1]
    assert rec2["first_say_audio_ms"] is None
    assert rec2["first_real_audio_ms"] is None
    assert rec2["full_answer_ms"] is None
    assert rec2["agent_turn_id"] == "run-0005"


async def test_run_wavs_resets_timers_between_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = _FakePlayer(close_delay_s=0.01)
    monkeypatch.setattr(playback_mod, "make_player", lambda *a, **k: player)
    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.05, {"text": "First answer."}),
            (0.0, {"done": True, "turn": "run-0001"}),
        ],
        [
            (0.01, {"text": "Second answer."}),
            (0.0, {"done": True, "turn": "run-0002"}),
        ],
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        t1 = time.perf_counter()
        h1 = HeardTurn("First question", None, None, t1, t1, t1)
        client = _FakeSpeechClient(heard_queue=[h1])
        model = AgentChatModel(url=url)
        session = _make_session(client, model, wav=["q1.wav", "q2.wav"], fillers=_fillers())

        async def next_wav(
            wav_path: Path | str, chunk_ms: int = 100, **kw: Any
        ) -> HeardTurn:
            del wav_path, chunk_ms, kw
            if client.heard_queue:
                return client.heard_queue.pop(0)
            t2 = time.perf_counter()
            return HeardTurn("Second question", None, None, t2, t2, t2)

        monkeypatch.setattr(client, "hear_wav", next_wav)
        await session.run_wavs()
    finally:
        ws_srv.shutdown()

    r1, r2 = session.records
    assert r1["agent_turn_id"] == "run-0001"
    assert r2["agent_turn_id"] == "run-0002"
    assert r1["fillers"] == ["One moment."]
    assert r2["fillers"] is None
    assert r2["first_real_audio_ms"] is not None
    assert r2["first_real_audio_ms"] < r1["first_real_audio_ms"]
    assert r2["t_speech_end_wall"] >= r1["t_speech_end_wall"]


def test_agent_client_captures_turn_id_and_emits_turn_event() -> None:
    from langchain_core.messages import HumanMessage

    steps: list[list[tuple[float, dict[str, Any]]]] = [
        [
            (0.0, {"text": "Hello."}),
            (0.0, {"done": True, "turn": "sess-0042"}),
        ]
    ]
    ws_srv, _srv, url = _serve_frames(steps)
    try:
        events: list[dict[str, Any]] = []
        model = AgentChatModel(url=url, on_event=events.append)
        chunks = [c.content for c in model.stream([HumanMessage("hi")]) if c.content]
    finally:
        ws_srv.shutdown()

    assert chunks == ["Hello."]
    assert model.last_turn_id == "sess-0042"
    assert events == [{"event": "text"}, {"event": "turn", "turn": "sess-0042"}]


def test_chunk_ignores_prefix_flag_directly() -> None:
    session = _make_session(_FakeSpeechClient(), play=False)
    push = session._chunk(None)
    session._current_is_prefix = True
    push(b"\x01\x00" * 100)
    assert session._answer_started is None
    assert session._say_started is None

    session._current_is_prefix = False
    push(b"\x01\x00" * 100)
    assert session._answer_started is not None
