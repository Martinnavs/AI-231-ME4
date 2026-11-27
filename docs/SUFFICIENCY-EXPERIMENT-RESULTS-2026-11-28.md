# Sufficiency verdict on the production model (Gemini Flash-Lite), dev split, 2026-11-28

Verdict through `gemini_json_request` with `SUFFICIENCY_SCHEMA`; the document lane stayed on Qwen. 50 dev rows, today=2026-11-27, `--verdict-backend gemini` (report `model=` still prints the Qwen lane model; verdict tokens show 0 because the counter only sees the Qwen relay; about 100 Gemini calls in all).

| mode | needs-web recall | stale | missing_half | conflict | controls over-trigger | verdict p50 / p95 |
|---|---|---|---|---|---|---|
| gold | 34/36 (94%) | 15/15 | 13/14 | 6/7 | 0/14 | 0.80 / 0.97 s |
| lane | 33/36 (92%) | 15/15 | 13/14 | 5/7 | 1/14 (imp-059) | 0.80 / 0.95 s |

Misses: imp-027 (missing_half judged sufficient, both modes), imp-042 (conflict), imp-040 (conflict, lane), imp-059 (control judged stale: "source is for 2027, question for 2026").
Reading: Qwen's dev numbers hold on the production model (stale 15/15 via the date fast path, 10-11 rows need no model call). The verdict is a hair weaker on conflict and shows one control over-trigger. Latency is about 0.8 s p50, in line with the Qwen measurement. Dev only; the frozen test half was not run on Gemini.

## Frozen test half on Gemini Flash-Lite (single run, no tuning afterwards)

50 test rows, same setup as the dev run above.

| mode | needs-web recall | stale | missing_half | conflict | controls over-trigger | verdict p50 / p95 |
|---|---|---|---|---|---|---|
| gold | 35/37 (95%) | 16/17 | 11/11 | 8/9 | 0/13 | 0.83 / 1.06 s |
| lane | 34/37 (92%) | 16/17 | 11/11 | 7/9 | 0/13 | 0.81 / 0.99 s |

Misses: imp-069 (stale, no lane answer, both modes), imp-100 (conflict, both modes), imp-036 (conflict, lane only). Qwen on the same half (gold): 34/37, controls 0/13. The production model matches or beats it on the frozen set; the known weak spot is still confident undated conflict facts.
