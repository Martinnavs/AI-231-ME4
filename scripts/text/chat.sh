#!/usr/bin/env bash
# chat.sh: the terminal chat against the agent server that up.sh started (TEXT_PORT_AGENT, default 18781).
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
exec uv run --offline python -m text_chat --url "ws://127.0.0.1:${TEXT_PORT_AGENT:-18781}" "$@"
