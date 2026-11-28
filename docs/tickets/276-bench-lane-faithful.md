# Ticket 276: make the MMLongBench-Doc lane production-faithful (catalog fallback + file_has), re-baseline dev

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). Free: Qwen only, NO Gemini/web. Dev split only. Touch ONLY
`scripts/bench_mmlongdoc.py`, `tests/test_bench_mmlongdoc.py`, `docs/MMLONGDOC-BENCH.md`. No change to ask_documents,
agent_server, readers, ingest, sufficiency. Commit explicit paths only. Do not run the full root suite (run the bench test file).

## Why (found by research agent 281, confirmed by the orchestrator in the code)
Production (`agent_server.py:836-875`) calls `ask_documents` with `catalog=_catalog` (the file list), `file_has=_file_has`
(a whole-file token check via `docs_enumerate`), `card`, `web`, `now`. Our bench runner (`scripts/bench_mmlongdoc.py`
`cmd_run`, the `ask` closure) passes `catalog=None`, `file_has=None`, `allow_fallback=False`. Consequences:
1. No fallback pass (`ask_documents.py:1295-1345`): when the first retrieval yields no usable answer, production asks the
   model to pick a file or search from the catalog and retries once. The bench never does, and 76 of 109 dev rows are `none`.
2. `file_has=None`: a question key token absent from the retrieved passages (`key_absent`) cannot be rescued by the
   whole-file check, so correct answers are rejected (8 of the 13 `check_failed` rows). The P6 finding in
   `docs/POTENTIAL-IMPROVEMENTS-PARKED.md` may be partly a harness artifact; this ticket measures how much.
So the 12.9% baseline understates the real lane. Everything compared later (A/B 269, fixes) must use the faithful lane.

## Do
1. In `cmd_run`: pass `catalog` = an async function returning the corpus file names (the same list `file_names` holds;
   check `docs_status`/`parse_docs_status` in `doc_route` for the shape production's `_catalog` returns and match it),
   `file_has` = an async function mirroring `agent_server._file_has` using the in-process `DbIndex`
   (`index.enumerate_files(token, limit=100)` returns rows; see `src/voice_agent/mcp_servers/docs.py:243` for how the
   tool wraps it, and reproduce the same fail-open rules: exception -> True, 100 rows -> True, match on `rel_key`),
   and `allow_fallback=True`. Keep `web=None` (docs-only). Keep `card` as now.
2. Add a flag `--lane-mode faithful|bare` (default `faithful`; `bare` = the old 272 behaviour, `catalog=None,
   file_has=None, allow_fallback=False`) and record `lane_mode` and `fallback` (from `DocResult.fallback`) in each run row.
   The fallback adds model calls: record `calls`. Resumable as before; tags differ so files do not collide.
3. Tests: with fake doc functions, `faithful` passes catalog/file_has/allow_fallback to the lane (spy on the call), `bare`
   does not; `_file_has` fail-open cases (exception, 100 rows, truncated); run rows carry lane_mode.
4. Run: `run --split dev --tag dev-faithful --lane-mode faithful` (Qwen answerer, all 109 dev rows), then
   `score --tag dev-faithful`, then the diagnosis script on the new run if it takes `--tag` (read its CLI; if it does not,
   do not modify it, just report that). Update `docs/MMLONGDOC-BENCH.md` (faithful is the default; the 272/273 numbers were
   `bare`).

## Final message
Table: bare vs faithful on the same 109 dev rows: accepted / check_failed / none counts, headline in-scope accuracy,
overall, official F1, unanswerable accuracy, by format, by source, cited-file == doc_id rate, mean calls and seconds per
question; the number of rows where fallback fired and how many of those ended accepted and correct; the rows that
changed status (list a few); the number of `check_failed` that disappeared; whether any `unanswerable` row became wrong.
State plainly whether the old baseline was an artifact and by how much. No other changes.
