"""Text mode of the voice agent: a terminal chat over the agent server's WebSocket.

No microphone, speech server, speaker or Pi: the agent server (router, tools, prompts, lookups)
is the one the voice stack runs; this package is only its client. `chat_ui` is vendored from
dsh-tui (see NOTICE.md).
"""
