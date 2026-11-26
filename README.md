# ME4 Pi experiment results: agent-server turn logs (2026-11-25 and 2026-11-26)

One row per turn from the Pi agent server (`pipeline/out/agent_server/*.jsonl`), copied unchanged. Code under test: branch `qwen/ingest`. The written reports (REPORT.md to REPORT5.md, turns*.json) and the findings note are on `qwen/ingest` under `.scratch/pi-check/` and `docs/ME4-VOICE-SESSION-FINDINGS-2026-11-26.md`.

Corpus: 44-file zip, then the 48-file `corpus48.zip` (13,898 chunks). Hardware: Raspberry Pi 4 (4 GB), Gemini through the stack. Filenames are the Pi's clock at server start (the Pi clock was wrong until 2026-11-26 about 12:20 local, so the first three dates are approximate).

| file | rows | what it is | code |
|---|---|---|---|
| 20261125-143851 | 5 | first text stack, IPv4 shim (offline-message turns) | 47817d7 |
| 20261125-144041 | 7 | first text stack, five timed turns with shim | 47817d7 |
| 20261125-175741 | 10 | ten-turn re-run (REPORT2) | dfc068f |
| 20261126-122904 | 16 | 16-turn text rehearsal on 48 files (REPORT3) | 324cd10 |
| 20261126-123811 | 16 | re-check of fix 234 (REPORT4) | 8dbc4e3 |
| 20261126-124318 | 32 | cold-start check, start 1, two passes (REPORT5) | 8dbc4e3 |
| 20261126-124505 | 16 | cold-start check, start 2 | 8dbc4e3 |
| 20261126-124628 | 16 | cold-start check, start 3 | 8dbc4e3 |
| 20261126-124816, 20261126-124937 | 2 each | first-question-at-port-open probes (early.sh) | 8dbc4e3 |
| 20261126-173512 | 60 | the owner's spoken session, HARNESS=pi, 40-question list | 8dbc4e3 |
| 20261126-181624 | 12 | replay of 12 utterances from that session on the latest code | 8cb8f94 |

Fields worth reading: `route` (docs, docs_miss, docs_general, routed, chain, fast, backchannel), `docs` (accepted, none, check_failed, timeout), `question`, `calls[].result_text` (retrieved passages), spoken text.

Notes:
- The planted decoy "Wi-Fi password" from the corpus appears only inside retrieved passages (`calls[].result_text`); it was never spoken in any answer.
- A key and secret scan of these files found nothing; no keys or tokens are written to these logs.
- Turn text in file `20261126-173512` is what the Pi heard (Parakeet with the 325-name vocabulary), so it contains recognition errors on purpose of the experiment.
