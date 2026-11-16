"""`python -m text_chat`: chat with the running agent server in the terminal.

The history the server sees is kept here, as the voice client keeps it (the last few exchanges,
plain text). The session token comes from $LAN_SPEECH_TOKEN or the file `scripts/text/up.sh`
wrote; it is sent as a header and never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from websockets.exceptions import WebSocketException

from text_chat.chat_ui import ChatUI
from text_chat.client import AgentClient, NoticeListener, render_turn

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "ws://127.0.0.1:18781"
DEFAULT_TOKEN_FILE = ROOT / "out/text/token"
BYE = "\\bye"


def load_token(path: Path) -> str | None:
    if env := os.environ.get("LAN_SPEECH_TOKEN"):
        return env.strip()
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def history_for(history: list[dict[str, str]], user: str, pairs: int) -> list[dict[str, str]]:
    """The last `pairs` exchanges and the new user message."""
    return (
        [*history[-2 * pairs :], {"role": "user", "content": user}]
        if pairs
        else [{"role": "user", "content": user}]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("TEXT_AGENT_URL", DEFAULT_URL))
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--history", type=int, default=4, help="exchanges sent with each turn")
    parser.add_argument("--no-notices", action="store_true", help="skip finished-job notices")
    args = parser.parse_args(argv)
    token = load_token(args.token_file)
    client = AgentClient(args.url, token)
    notices = None if args.no_notices else NoticeListener(args.url, token)
    if notices is not None:
        notices.start()
    ui = ChatUI(
        agent_cursor="agent",
        banner_title="Voice agent: text mode",
        disclaimer="The agent may be wrong. Check anything that matters. \\bye or Ctrl+D to leave.",
        show_logs=True,
        max_messages=1000,
    )
    history: list[dict[str, str]] = []
    while True:
        while notices is not None and not notices.notices.empty():
            note = notices.notices.get()
            ui.add_message("assistant", str(note.get("text", "")))
            notices.ack(note)
        user = ui.get_user_input()
        if user == BYE:
            return 0
        if not user.strip():
            continue
        try:
            result = render_turn(client.frames(history_for(history, user, args.history)), ui)
        except (OSError, WebSocketException) as exc:
            ui.stop_thinking()
            ui.notice(f"Could not reach the agent server at {args.url}: {exc}", level="error")
            ui.add_message("system", f"Turn failed: {exc}")
            continue
        history += [{"role": "user", "content": user}]
        if result.text:
            history += [{"role": "assistant", "content": result.text}]


if __name__ == "__main__":
    sys.exit(main())
