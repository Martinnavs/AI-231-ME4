# Text mode (branch `text-only`)

The voice agent without the audio: the same agent server (router, fast path, tools, grounded lookups, delegate, prompts) driven from a terminal chat. No microphone, speech server (STT/TTS), speaker or Pi. This is the dev branch for the machine exercise, where the speech side is already settled.

```
make text-up              # router + agent server on loopback (ports 18089, 18781), state in out/text/
make text                 # the chat UI (`\bye` or Ctrl+D leaves; PgUp/PgDn scroll)
make text-down
make text-up NO_ROUTER=1  # Gemini answers every turn; no GGUF model needed
```

Needs: the root and pipeline venvs (`uv sync --group laptop && make pipeline-sync`), a Gemini key at `~/.secrets/google-ai-studio/api-key.txt` (read by the code, never printed), and for the router the GGUF at `~/work/gguf-tools/models/$ROUTER_MODEL/gguf/` (`make setup-router`). The venue is the University of the Philippines Cebu (a pin, no IP lookup); change it with `VOICE_AGENT_HOME_COORDS="lat,lon"`.

## What is where

| Piece | File | Notes |
|---|---|---|
| Chat UI | `src/text_chat/chat_ui/` | vendored unchanged from [dsh-tui](https://github.com/Martinnavs/dsh-tui) (from onit's `ui/text.py`, Apache-2.0, `NOTICE.md`); skipped by ruff and mypy |
| Protocol client | `src/text_chat/client.py` | `render_turn` draws one turn's frames; `AgentClient` is the WebSocket; `NoticeListener` collects finished-job notices |
| Entry point | `src/text_chat/cli.py` | keeps the history the server sees (last 4 exchanges, like `doc_chat --history 4`) |
| Launch | `scripts/text/{up,down,chat}.sh` | own token (`out/text/token`, mode 0600), own delegate state, ports 18xxx so the voice stack can run beside it |

The agent server is unchanged: a turn is `{"messages": [...]}` in, `{"text"}` sentences, `{"event": "tool"}` and `{"done"}` out (`agent_server.AgentServer.handle`).

## Differences from the voice loop

- Sentences arrive as the server cuts them (the voice client speaks each); the UI joins them with a space. Nothing is spoken, so `speakable.py` and the TTS text prep do not matter here, but the server's spoken-style filters (no markdown, short answers) still apply, because the prompt says the answer is spoken.
- Tool calls show as a name only (the server sends the name, not the arguments or result).
- Other events (`open_url`, `open_app`) are shown as a notice line, not acted on.
- Finished-job notices are shown before the next prompt, not spoken when you are idle.
- No echo, VAD, addressee gate, barge-in or filler audio; the endpoint-silence and STT settings do not apply.
- The Pi tools (`--pi-url ""`) and the apps server are not started: speaker volume, device status and `open_app` for a built page are unavailable.

Tests: `tests/text_chat/` (frames drawn on a fake UI, the history, the token, a real WebSocket round trip against a stand-in server speaking the protocol). Checked by hand against the real agent server: "what time is it" through the pty chat.
