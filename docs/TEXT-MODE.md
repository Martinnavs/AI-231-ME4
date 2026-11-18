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

Since the agent roadmap (steps 1-3 and F2a; `docs/AGENT-ASSESSMENT.md` section 12):

- The changeable-fact note (`CHANGEABLE_NOTE`) now also reaches hedged turns: `AgentServer.stamped` appends it, then the `[Now: ...]` line, to the last user message both hedged Gemini requests get.
- Second-look results are written to `out/agent_server/checks/<run>.jsonl`, not to the turn file.
- The text client (`src/text_chat/client.py`) sends a `session` string with every request (one `uuid4().hex` per client process); the server keeps at most 64 characters of it (`trace.SESSION_CAP`).
- Every turn is traced as one row in `out/agent_server/<run>.jsonl`: `docs/TURN-TRACE.md`.

## Chatting about your documents

Drop `.md`, `.txt`, `.csv`, `.pdf` or `.docx` files in `~/agent_sandbox/docs` and ask about them in the same text chat; new and edited files count from the next question, no restart. The demo server for this is on port 8782 (the stack's 8781 stays as it is):

```
.scratch/agent-roadmap/live/demo_server.sh start
uv run python -m text_chat --url ws://127.0.0.1:8782
.scratch/agent-roadmap/live/demo_server.sh stop
```

A document question is answered on a new route, `route: "docs"`: `As per <file>, <section | p. N>, modified <date>: <answer> "<verbatim quote>"`, or `The documents don't say. Want me to look it up on the web?` (a short "yes" then runs a web lookup). The turn row gets `docs` (the outcome of the document check: `accepted`, `none`, `check_failed`, `error` or `timeout`), `doc_s` (seconds spent on the documents) and, when a check rejected the model's answer, `rejected` (`docs/TURN-TRACE.md`). Setup, the checks, settings and limits: `docs/RAG.md`.

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

## Second opinions

`make text-up` allows 100 checks (`VOICE_AGENT_VERIFY_MAX`): after a reply about something that changes, a second grounded look runs in the background and a different answer arrives as a notice before your next prompt (docs/LIVE-AGENT.md, the second opinion). `VOICE_AGENT_VERIFY_MAX=0 make text-up` turns it off; the count lives in `~/.local/state/voice_agent/verify_used.json`.

