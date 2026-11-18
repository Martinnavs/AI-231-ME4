# Turn trace

One JSON row per agent-server turn in `out/agent_server/<run>.jsonl` (the file stem is the run id),
written by `AgentServer.handle` (`pipeline/src/voice_pipeline/agent_server.py`) with the helpers in
`pipeline/src/voice_pipeline/trace.py`. This is the shared contract for everything that reads turn
rows (evaluators, the agent benchmark). All new fields are additive: the keys the row had before
are unchanged.

## Example

A hedged turn the model won: the router's lookup was cancelled, the model's ran. Produced by
`serve_turn` in `pipeline/tests/test_trace.py` (the real `AgentServer.handle` over a real local
WebSocket, fake agents and lookups); `test_the_documented_example_has_the_keys_of_a_real_row`
fails when the keys here drift from a real row.

<!-- example-row -->
```json
{
  "time": "08:27:38",
  "question": "Who is the CEO of Nvidia?",
  "calls": [
    {
      "tool": "web_answer",
      "args": {
        "query": "who is the ceo of nvidia?"
      },
      "by": "router",
      "t0": 0.0,
      "s": 0.0,
      "status": "cancelled"
    },
    {
      "tool": "web_answer",
      "args": {
        "query": "ceo of nvidia"
      },
      "by": "model",
      "t0": 0.0,
      "s": 0.0,
      "status": "ok",
      "result_sha": "193d067f2e80",
      "result_chars": 53,
      "result_text": "As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
      "sources": [
        "CNN",
        "FT"
      ],
      "only_one_source": false
    }
  ],
  "cancelled": false,
  "run": "20261118-120000",
  "turn": "20261118-120000-0001",
  "session": "3f2a9c1e5b7d4a60",
  "route": "routed",
  "tail": [
    "note",
    "now"
  ],
  "tokens": {
    "requests": 2,
    "prompt": 180,
    "completion": 0,
    "cached": 0
  },
  "ledger": {
    "path": "/tmp/tmp206cdr9p/ledger.jsonl",
    "from": 0,
    "to": 52,
    "n": 1,
    "rows": [
      {
        "time": "2026-11-18 12:00:03",
        "month": "2026-11",
        "searches": 1,
        "tokens": 412,
        "seconds": 1.84,
        "query": "ceo of nvidia"
      }
    ]
  },
  "status": "ok",
  "text": "As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
  "hedge_winner": 1,
  "hedge_requests": 2,
  "hedge_offset_s": 0.0,
  "hedge_total_s": 0.0005942379939369857,
  "router_s": 0.1,
  "router_lines": "web_answer|who is the ceo of nvidia?"
}
```

The ledger rows in the example have the shape the MCP child writes (`gemini_grounding.Ledger`: `time`, `month`, `searches`, `tokens`, `seconds`, and `query`, plus `sources`, `check`, `forced` or `failed` when they apply); the test's own ledger rows are fixtures, so the values are illustrative and only the keys of the row, `ledger`, `tokens` and `calls` are checked.

## Row

| Key | Meaning |
|-----|---------|
| `time`, `question`, `calls`, `cancelled`, `unchecked`, `dropped_payload`, `dropped_sentences`, plus the turn result (`status`, `text`, `first_call_s`, `first_text_s`, `end_s`, `routed`, `router_s`, `router_lines`, `race`, `hedge_*`, `cited_by_server`, ...) | as before this trace existed |
| `run` | the log file's stem, e.g. `20261118-075023` |
| `turn` | `<run>-<n:04d>`, n counts handled requests of this server process from 1. The `{"done": true, "turn": <id>}` event carries it; a cancelled turn has a row too |
| `session` | the client's `session` string from the request, cut to 64 characters; absent when missing, empty or not a string (the value is untrusted) |
| `route` | the branch of `handle` that took the turn: `cancel_chain`, `backchannel`, `fast`, `chain` (a running research/build task), `routed` (the router branch; with a hedged agent the model may still win, see `hedge_winner`), `gemini` (no router, or the router branch was skipped), `docs` (the document route: the user named their documents, or the implicit gate passed and `doc_answer` accepted; `docs/RAG.md`) |
| `docs`, `doc_s`, `rejected` | the document check of the turn, as top-level keys of the turn result (`doc_route.handle_docs_turn`, `handle_implicit_docs_turn`). `docs` is the `DocAnswer` status: `accepted` (a checked, cited answer was spoken), `none` (the model said NONE, no passages, or the folder is empty), `check_failed` (the model's answer failed a check), `error`, or `timeout` (the 2.5 s budget ran out). `doc_s` is the seconds spent searching and checking. `rejected` is `{answer, quote, failed}` and is present only when a check rejected the answer; `failed` is `quote`, `names_numbers`, `quote_length`, `overlap` or `directive`. On an explicit document turn (`route` `docs`) all three keys are set (`rejected` only when there was one). On an implicit turn that falls through to `routed` or `gemini`, only `doc_s` is added, because the document attempt is not spoken. When the router's web lookup raced the documents (`doc_race`), the same three keys sit inside `race` instead |
| `tail` | the parts appended to the last user message, as built for this turn: `chain` (chain state), `case_f` (the Case F protocol), `note` (the changeable-fact note), `now` (the `[Now: ...]` line). A hedged turn carries `note` (when the last user message is a changeable question, same test as `steer`) then `now` (`AgentServer.stamped`); the other parts come from `AgentServer.steer`. On a router-won hedged turn `tail` is what the Gemini requests actually received: both hedged requests start before the router decides, so they got the note even though the router answered (`note_reached` is n/a on router-won turns) |
| `tokens` | `{requests, prompt, completion, cached}`: the change in every model request's `TurnTap.usage` during the turn, hedge losers included (a loser still running when the turn ends is only partly counted). Needs `Agent(usage=True)`, which `run()` sets; `completion` includes reasoning tokens; fakes without a tap count 0 |
| `ledger` | see below |
| `calls` | one span per tool call, in the order the calls started |

## Call spans (`calls[i]`)

Existing keys: `tool`, `args`, `cached`, `raw_task`, `intent`, `slot`, `result` (delegate), unchanged.

| Key | Meaning |
|-----|---------|
| `by` | `model`, or the server's own `tool_call_id`: `router` (router calls, the doc race, retries, the second look's tool), `fast-path` |
| `t0` | seconds from the start of the turn to the start of the call |
| `s` | seconds the call took |
| `status` | `ok`; `failed` (it raised, or its result is a failure text such as "couldn't find" or `{"ok": false}`); `cancelled` (the task was cancelled, e.g. the router's lookup when the model wins the hedge); `joined` (the existing `cached` reuse: the call got the turn's earlier lookup instead of running one) |
| `result_sha`, `result_chars` | first 12 hex characters of the sha256 of the result text, and its length; absent when the call produced no result |
| `result_text` | the result text, at most 2,000 characters, for the lookup tools (`web_research`, `web_answer`, `wiki_summary`), `search_docs` and `docs_status` only. For `search_docs` it is the raw passages (`{source, id, section, page, file_date, text, score, coverage}`), not the `<untrusted_document>` wrapping the model sees. The document route's own calls are spans with `by` `router` |
| `sources`, `only_one_source` | the publishers the lookup names and whether it found only one site, parsed by `agent_tools._lookup_sources`; lookup tools only, when the result carries them |

Hedge losers never run a tool handler (`hedge.gated_tools` answers `superseded`), so every span
belongs to the winner or to the router. The `smart_home` and music tools write their own
`{"intent", "slot"}` entries and are not wrapped by `_recorded`: such an entry gets `by`, `t0`,
`s`, `status` only when the server called it (`router`, `fast-path`), not when the model did.

## Ledger slice (`ledger`)

`{path, from, to, n, rows}`: the web-answer ledger (`search_count.LEDGER`, a JSON-lines file) is
measured in bytes when the turn starts and when its model/router work ends, and `rows` are the
rows appended between the two offsets (at most 10 kept, `n` is how many were there). Rows written
before the turn are not in it. A row with a `check` key was written by a background second look,
which can finish during the next turn: it belongs to that earlier turn's `checks/` row, not to
this turn, so a reader that counts searches for a turn skips rows that have `check`.

The MCP child (the writer of the ledger rows) does not inherit the server's environment
(`mcp.client.stdio.get_default_environment` passes only HOME, PATH and a few more), so when
`WEB_ANSWER_LEDGER` is set to a non-default path the child and the server would not share a file.
`path` records the file the server read; compare it with the child's when that variable is used.

## Second looks (`checks/<run>.jsonl`)

`AgentServer.check_in_background` writes one row per finished second look to
`out/agent_server/checks/<run>.jsonl` (a subdirectory, so `scripts/output_audit.py`'s `*.jsonl`
glob does not pick it up): `{time, run, turn, question, verdict, s, ledger}`. `turn` is the turn
that was checked, `verdict` the first line of the answer (60 characters), `s` its duration, and
`ledger` its own slice (same shape as above). The turn file no longer gets `second_opinion` rows.
