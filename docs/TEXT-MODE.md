# Text mode (branch `text-only`)

The voice agent without the audio: the same agent server (router, fast path, tools, grounded lookups, delegate, prompts) driven from a terminal chat. No microphone, speech server (STT/TTS), speaker or Pi. This is the dev branch for the machine exercise, where the speech side is already settled.

```
make text-up              # scripts/stack/up.sh text: router + agent server on loopback (ports 8089, 8781), logs in out/stack/
make text                 # the chat UI (`\bye` or Ctrl+D leaves; PgUp/PgDn scroll)
make text-down
make text-up NO_ROUTER=1  # Gemini answers every turn; no GGUF model needed
```

`up.sh text` is the stack's laptop mode without the cable: `load_env` skips the Pi checks (`TEXT=1`), only the router and the agent server start, and the agent runs with `--pi-url ""`. It uses the stack's ports and session token (`~/.config/lan_speech_token`), so `make up` and `make text-up` cannot run together; `make text-down` stops by PID file with the same checks as `make down`.

Needs: the root and pipeline venvs (`uv sync --group laptop && make pipeline-sync`), a Gemini key at `~/.secrets/google-ai-studio/api-key.txt` (read by the code, never printed), and for the router the GGUF at `~/work/gguf-tools/models/$ROUTER_MODEL/gguf/` (`make setup-router`). The venue is the University of the Philippines Cebu (a pin, no IP lookup); change it with `VOICE_AGENT_HOME_COORDS="lat,lon"`.

## What is where

| Piece | File | Notes |
|---|---|---|
| Chat UI | `src/text_chat/chat_ui/` | vendored unchanged from [dsh-tui](https://github.com/Martinnavs/dsh-tui) (from onit's `ui/text.py`, Apache-2.0, `NOTICE.md`); skipped by ruff and mypy |
| Protocol client | `src/text_chat/client.py` | `render_turn` draws one turn's frames; `AgentClient` is the WebSocket; `NoticeListener` collects finished-job notices |
| Entry point | `src/text_chat/cli.py` | keeps the history the server sees (last 4 exchanges, like `doc_chat --history 4`) |
| Launch | `scripts/stack/{up,down}.sh text`, `scripts/text/chat.sh` | the stack's scripts, ports and token; text mode is a branch of them, not a second launcher |

The agent server is unchanged: a turn is `{"messages": [...]}` in, `{"text"}` sentences, `{"event": "tool"}` and `{"done"}` out (`agent_server.AgentServer.handle`).

## Differences from the voice loop

- Sentences arrive as the server cuts them (the voice client speaks each); the UI joins them with a space. Nothing is spoken, so `speakable.py` and the TTS text prep do not matter here, but the server's spoken-style filters (no markdown, short answers) still apply, because the prompt says the answer is spoken.
- Tool calls show as a name only (the server sends the name, not the arguments or result).
- Other events (`open_url`, `open_app`) are shown as a notice line, not acted on.
- Finished-job notices are shown before the next prompt, not spoken when you are idle.
- No echo, VAD, addressee gate, barge-in or filler audio; the endpoint-silence and STT settings do not apply.
- The Pi tools (`--pi-url ""`) and the apps server are not started: speaker volume, device status and `open_app` for a built page are unavailable.

Tests: `tests/text_chat/` (frames drawn on a fake UI, the history, the token, a real WebSocket round trip against a stand-in server speaking the protocol). Checked by hand against the real agent server: "what time is it" through the pty chat.

## The changeable-question gate

Run `make setup-gate` once (network, about 30 s to a few minutes, scratch torch in a throwaway venv) to build `out/models/eg-e5/`: the EverGreen classifier that decides whether a question needs a lookup (docs/CITATION-TAXONOMY.md). Without it the regex answers, so nothing breaks. The model runs inside the MCP child that serves `web_answer`; the pipeline's own note for changeable questions (`call_policy.changeable_question`) still uses the regex, because the pipeline venv has no `tokenizers`.

