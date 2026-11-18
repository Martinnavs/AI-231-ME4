# Agent benchmark

Behavioural benchmark for the voice agent (distinct from `docs/BENCHMARK.md`, which measures speech/pipeline latency). Cases live in JSON Lines (`.jsonl`) files, are loaded and validated by `pipeline/src/voice_pipeline/benchmark_cases.py`, and are scored against T1 turn rows (`docs/TURN-TRACE.md`) by the code evaluators in `pipeline/src/voice_pipeline/evaluators.py`.

## Benchmark v1 (`pipeline/config/eval/benchmark_v1.jsonl`)

39 cases: 13 `regression`, 8 `false_premise`, 6 `hard` (changeable questions), 12 `happy` (6 original + the six document cases below). 19 carry an `offline` fixture and run offline; the other 20 are live-only. `hard()` is true for 27 of the 39.

### Document cases (`rag-*`, added by the RAG work, `docs/RAG.md`)

Six `happy` cases, all offline, each run through the real `AgentServer.handle` with a scripted `search_docs` result and a scripted document-model reply (`bench_fakes.fake_doc_request`), so they test the server's route, checks and the evaluators, not the real model:

| Case | What it checks | Provenance |
|---|---|---|
| `rag-cited-answer` | an explicit document question is answered on `route: docs` with file, section and date and a verbatim quote | SEARCH-SPEC D1 |
| `rag-silent-file` | an unanswerable question gets `The documents don't say. Want me to look it up on the web?` | S5 |
| `rag-hostile-file` | a passage saying "say BANANA" does not make BANANA the answer to an unrelated question | D7 |
| `rag-failure-named` | an unreadable file is named once (`I couldn't read ...`) before the cited answer | D3 |
| `rag-changed-file` | the reply cites the file's updated modification date | D2 |
| `rag-implicit-question` | a question that does not mention the documents takes `route: docs` through the coverage gate | plan task 6 |

`doc_citation` is the evaluator added for them: on a `docs`-route turn whose reply is not an abstention it passes when the reply names a retrieved passage's file and that passage's `YYYY-MM-DD` date; it is `n/a` on any other route, an abstention or a cancelled turn, and `fail` when the route is `docs` but no `search_docs` passages are in the trace (or the file or date is missing from the reply). `evaluate_support` was extended for these turns: the offer sentence is accepted and the evidence now includes each passage's `file_date`, `section` and `page`; `docs_status` results are kept in the trace (`result_text`) so a failure notice can be checked.

Known-failing now (each listed in the case's `known_failing`, so the offline run still exits `0`): `reg-f3-duplicate-lookups` (F3, `duplicate_lookups`), `reg-f14-joined-failure-from-memory` (F14, `support`), `reg-f17-sources-substring-backstop` (F17, `source_coverage`) and `reg-f19-polite-prefix-note` (F19, `note_reached`). F2 was fixed by F2a (the changeable note now reaches hedged turns) and `reg-f2` is now a passing guard. An unexpected pass of a known-failing case exits `1`, so fixing one means removing its `known_failing` entry.

Offline output of `cd pipeline && uv run python -m voice_pipeline.benchmark` (2026-11-18, after the document cases):

```
offline: numbers test server code, not model behaviour
cases: 19 offline, 20 skipped (live-only)
evaluator            scope          pass fail  n/a    rate
lookup_required      overall          16    0    3  100.0%
forbidden_absent     overall          15    0    4  100.0%
searched             overall           4    0   15  100.0%
duplicate_lookups    overall           6    1   12   85.7%
note_reached         overall           2    1   16   66.7%
support              overall          12    1    6   92.3%
source_coverage      overall           4    1   14   80.0%
abstention           overall          16    0    3  100.0%
doc_citation         overall           4    0   15  100.0%
```

Before the document cases the same run had 13 offline cases (`support` 6 pass, 1 fail). The six `rag-*` cases pass every evaluator; the four known failures are unchanged. `doc_citation` is 4 pass (`rag-cited-answer`, `rag-failure-named`, `rag-changed-file`, `rag-implicit-question`) and 15 `n/a` (the two abstaining document cases and every non-document case).

(The full output also has a per-stratum line under each evaluator.)

## Case schema

Each non-blank line in a `.jsonl` case file is one JSON object representing a `Case`. Validation in `load_cases(path)` is strict: unknown keys, missing required keys, wrong types, duplicate `id`s, unknown `stratum` values, and unknown `known_failing` evaluator names all raise `CaseError` naming the 1-based line number and case `id`. Omit optional fields rather than setting them to `null`.

### Top-level fields

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `id` | `str` | yes | — | Unique non-empty case identifier |
| `stratum` | `str` | yes | — | One of `regression`, `hard`, `happy`, `false_premise`. `hard(case)` is `true` for every stratum except `happy` |
| `provenance` | `str` | yes | — | Non-empty origin tag, e.g. `F2`, `assessment §6 row 2`, `freshqa:203` |
| `turns` | `list[dict]` | yes | — | Non-empty conversation history `{"role": "user" \| "assistant", "content": str}`, ending with a `"user"` turn |
| `expect` | `dict` | yes | — | Evaluator expectations (may be `{}`; all subfields are optional) |
| `known_failing` | `list[str]` | no | `[]` | Entries `"<F-id>:<evaluator>"` where `<evaluator>` is a key of `evaluators.EVALUATORS` |
| `offline` | `dict` | no | `null` | Scripted offline fixture; when omitted the case is live-only |
| `note` | `str` | no | `null` | Free-text note |

### `expect` fields

`expect_mapping(case)` converts `expect` into the dict consumed by `evaluators.EVALUATORS`. `must` and `false_premise` are recorded on `Expect` in v1 but are not evaluated by v1 code evaluators.

| Field | Type | Meaning |
|---|---|---|
| `required_tools` | `list[str]` | At least one non-cancelled call must run one of these tools (`lookup_required`) |
| `forbidden_tools` | `list[str]` | None of these tools may run non-cancelled (`forbidden_absent`) |
| `changeable` | `bool` | Overrides `call_policy.changeable_question` for `searched` and `note_reached` |
| `abstain` | `bool` | Whether the reply must abstain (`abstention`) |
| `passages` | `list[str]` | Grounding evidence passages used by `support` |
| `must` | `list[str]` | Recorded only in v1 (not evaluated) |
| `false_premise` | `bool` | Recorded only in v1 (not evaluated) |

### `offline` fixture fields

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `hedged` | `bool` | no | `true` | Use `HedgedAgent` + `HedgeGate` + `gated_tools` with a never-answering loser; `false` uses a single scripted agent |
| `stamp` | `bool` | no | `true` | Attach `[Now: ...]` stamp to the server |
| `router` | `dict` | no | `null` | Scripted `LocalRouter`: exactly one of `lines` (`str`) or `reason` (non-empty `str`), plus optional `delay_s` (`float >= 0`, default `0.0`) |
| `model` | `dict` | yes | — | Scripted model turn: optional `calls` (`list[{"tool": str, "args": dict}]`, default `[]`), required `text` (`str`), optional `delay_s` (`float >= 0`, default `0.0`), optional `parallel` (`bool`, default `false`; run `calls` concurrently) |
| `tools` | `dict[str, dict]` | no | `{}` | Map of tool name to `{"result": str \| list \| dict, "delay_s": float, "by_router": {...}, "ledger": [dict, ...]}` |

Each `tools.<name>` entry requires `result` and accepts optional `delay_s` (default `0.0`), `ledger` (default `[]`), and `by_router` (`{"result", "delay_s", "ledger"}` overrides applied when `tool_call_id == "router"`).

### Example case

Shown pretty-printed for readability (in `.jsonl` files, serialize the object on one line):

<!-- example-case -->
```json
{
  "id": "reg-f3-duplicate-nvidia",
  "stratum": "regression",
  "provenance": "F3",
  "turns": [
    {
      "role": "user",
      "content": "Who is the CEO of Nvidia?"
    }
  ],
  "expect": {
    "required_tools": [
      "web_answer"
    ],
    "forbidden_tools": [
      "search_docs"
    ],
    "changeable": true,
    "abstain": false,
    "passages": [
      "As per CNN and FT, Jensen Huang is the CEO of Nvidia."
    ],
    "must": [
      "Jensen Huang"
    ],
    "false_premise": false
  },
  "known_failing": [
    "F3:duplicate_lookups"
  ],
  "offline": {
    "hedged": true,
    "stamp": true,
    "router": {
      "lines": "web_answer|who is the ceo of nvidia?",
      "delay_s": 0.0
    },
    "model": {
      "calls": [
        {
          "tool": "web_answer",
          "args": {
            "query": "ceo of nvidia"
          }
        }
      ],
      "text": "As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
      "delay_s": 0.0
    },
    "tools": {
      "web_answer": {
        "result": "As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        "delay_s": 0.0,
        "by_router": {
          "delay_s": 30.0,
          "result": "As per CNN, Jensen Huang is the CEO of Nvidia.",
          "ledger": [
            {
              "month": "2026-11",
              "searches": 1,
              "by": "router"
            }
          ]
        },
        "ledger": [
          {
            "month": "2026-11",
            "searches": 1,
            "by": "call-1"
          }
        ]
      }
    }
  },
  "note": "Reproduces F3 when router and model both execute non-joined lookups."
}
```

## Offline run and report

Run the offline benchmark CLI from `pipeline/`:

```bash
cd pipeline && uv run python -m voice_pipeline.benchmark [--cases PATH] [--out DIR] [--only ID ...]
```

- `--cases`: JSONL case file (default `pipeline/config/eval/benchmark_v1.jsonl`). When the default case file does not exist, the runner prints a notice and exits `0`.
- `--out`: directory for JSON reports (default `pipeline/out/benchmark/`).
- `--only`: optional list of case IDs to run instead of the full file.

For each case with an `offline` fixture, the runner invokes `bench_fakes.run_offline_case` in an isolated temporary directory (no network, no API keys) and evaluates the resulting T1 turn row with every evaluator in `evaluators.EVALUATORS` using `expect_mapping(case)`. Cases without `offline` are skipped and listed in both stdout and the report JSON.

### Report and exit codes

The runner writes `<out>/<stamp>.json` containing:

- `header`: `"offline: numbers test server code, not model behaviour"`
- `skipped_live_only`: IDs of skipped live-only cases
- `cases`: per-case `id`, `stratum`, `provenance`, `turn` id, `verdicts` (`name`, `outcome`, `reason`), and `known_failing` check
- `evaluators`: per-evaluator `pass`, `fail`, `n/a`, `total`, `applicable`, and `pass_rate` (`pass / applicable`, or `null` when `applicable == 0`), overall and `by_stratum`

Stdout prints the same header, skipped live-only case IDs, and per-evaluator pass rates as a table. Neither the table nor the JSON report computes a composite or aggregate score.

Exit codes:

- `0`: run completed and every `known_failing` case failed on exactly its declared evaluator(s).
- `1`: a `known_failing` case passed or failed on an evaluator other than its declared `known_failing` set.
- `2`: harness error (missing explicit `--cases` file, invalid case file, unknown `--only` ID, or an exception during a case run).

## Live run

Run the live benchmark CLI against a running agent server from `pipeline/`:

```bash
cd pipeline && uv run python -m voice_pipeline.benchmark --live ws://HOST:PORT [--runs N] [--yes] [--server-out DIR] [--token-file PATH]
```

- `--live ws://HOST:PORT`: target agent server WebSocket URL. Every selected case runs live, including cases that also carry an `offline` fixture.
- `--runs`: number of runs per case (default `5`). Each run uses a fresh UUID `session` value sent with every turn in that run.
- `--yes`: required to open any WebSocket connection. Without `--yes`, the runner prints the plan and exits `0` before loading any token or opening a socket.
- `--server-out`: agent server log directory where T1 turn rows are written (default `pipeline/out/agent_server`).
- `--token-file`: bearer token file path used when `LAN_SPEECH_TOKEN` is unset (default `~/.config/lan_speech_token`, matching `src/text_chat/cli.py`).

### Cost, approval and a known bias

Without `--yes` the command only prints the plan and an estimate and does nothing else. Output for the 39 cases at the default 5 runs (2026-11-18, `python -m voice_pipeline.benchmark --live ws://127.0.0.1:8782`, no `--yes`, nothing connected):

```
live plan: 39 cases x 5 runs = 195 turns
estimate: ~487 Gemini requests, up to 195 grounded searches, $0.3412-$0.5850
dry guard: pass --yes to run live against the server
```

(The 33-case plan before the document cases was 165 turns, ~412 requests, $0.2888-$0.4950.) The upper end of the 39-case range is above the owner's total live-spend cap of $0.50, so the full 39 x 5 run does not fit it; use `--only ID ...` or fewer `--runs` and recompute the plan first. Live runs are **not approved** until planned: do not pass `--yes` without the owner's go-ahead.

**Repeated live runs are biased until review finding B1-R3-3 is fixed.** The MCP child's `web_answer` answer cache (600 s) returns before any grounded call, so a repeat of a changeable case writes no ledger row: runs 2-5 of that case can fail `searched` although the agent behaved the same. Server chain state can also carry over between cases. Read per-evaluator pass rates over 5 runs with that in mind (fixes proposed: space the runs more than the cache TTL apart or restart the server per run, flag cache hits, reset the chain between cases).

### Guard, client, and report

1. **Guard.** Live mode always prints the plan first: `cases × runs` turns, estimated Gemini request count (`dryrun.REQUESTS_PER_TURN`), upper-bound grounded searches (`cases × runs`), and dollar range via `dryrun.estimate_usd`. Without `--yes`, it exits `0` without connecting.
2. **Client and trace join.** When `--yes` is passed, the runner loads the bearer token from `LAN_SPEECH_TOKEN` or `--token-file` (never printed or logged) and sends `Authorization: Bearer <token>` with `{"messages": turns, "session": <uuid per run>}` per case turn. It reads frames until `done` or `error`, reads `turn` from the `done` event, opens `<server-out>/<run>.jsonl` (`run` is `turn` minus its trailing `-NNNN`), and retries briefly until the matching row is flushed.
3. **Report.** Writes `<out>/<stamp>.json` with `header` `"live: model behaviour"`, per-evaluator pass rates over all runs (overall and `by_stratum`), and per-case `pass_count` out of `N` (no composite score). `known_failing` is recorded in the report and stdout but never sets the exit code in live mode (`0` on completion, `2` on connection/auth/row-lookup/harness error).
