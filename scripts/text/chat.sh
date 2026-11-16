#!/usr/bin/env bash
# chat.sh: the terminal chat against the agent server that `scripts/stack/up.sh text` started (port 8781, the stack's token).
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
exec uv run --offline python -m text_chat "$@"
