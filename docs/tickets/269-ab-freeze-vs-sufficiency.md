# Ticket 269 (PLAN): A/B, frozen release vs sufficiency branch, on an MMLongBench-Doc subset

Status: PLAN, not dispatched. Blocked on: 268 (clock) merged; bench/telemetry (265/266) merged into the
test branch; ideation tickets T3 (MMLongBench-Doc adapter + scorer bridge) and T5 (`--ocr` on `.zip`).
Owner's order: sufficiency work first, then MMLongBench-Doc; this plan is the bridge between them.

## Question
Is the sufficiency branch (flag on) better than the frozen release on a document benchmark it was not
built for, and does it cost anything (wrong supplements, latency, spend)?

## Honest framing (decided up front)
MMLongBench-Doc asks about static documents; almost nothing is "stale". The sufficiency check should
therefore rarely fire. This A/B measures **no regression + over-trigger cost** (and the effect of the
supplement on the 20.6% unanswerable questions, where a web sentence can turn a correct "not in the
documents" into a wrong answer). The **gain** is measured on the 100-row sufficiency set (already done:
`docs/SUFFICIENCY-EXPERIMENT-RESULTS-2026-11-28.md`). Report both; never claim the benchmark shows the gain.

## Arms (same corpus, same questions, same answerer and judge models, same Gemini key, same clock)
- **A** frozen tag `me4-rc-2026-11-27` (`1421c7c`).
- **C** `exp/sufficiency` + 268, `SUFFICIENCY=False` (isolates the bug fixes: compound split, schema argument).
- **B** `exp/sufficiency` + 268, `SUFFICIENCY=True`.
B vs C = the feature; C vs A = the fixes. Each arm runs from its own worktree against its own copy of the
ingested DB built from the same raw files.

## Test set
- MMLongBench-Doc pinned revision (record the commit/revision and row count; 1,082 vs 1,091 drift).
- Subset ~200 questions, stratified: text, table, scanned/OCR, chart-only (control, reported separately,
  not in the in-scope headline), and ~40 unanswerable. Seeded sampling (sha256 rule like
  `implicit-split:7`), split dev/test by the same hash rule: tune/debug on dev only, run the test half
  once per arm. Write the sampled ids and manifest to `eval/mmlongbench/` before any run.
- PDFs: only the documents the subset touches (about 25), ingested with `--ocr` after T5.

## Procedure
1. Offline text runs (laptop, no audio): each arm answers each question through the real agent path
   (`doc_chat` / agent server in docs-first mode, web allowed as in production). Per-turn trace from T1/T2:
   cited_file/pages, supplement fired, web calls, `t_first_real_audio`-style timing.
2. Scoring: T3's typed-answer extractor (local Qwen replacing GPT-4o) + the official `eval_score.py` rules;
   calibrate on 50 audited rows first (agreement >= 90%, kappa >= 0.80) or stop.
3. Paired comparison per question across arms: accuracy and unanswerable F1 with a bootstrap 95% CI
   (n=200 resolves only about +-7 points; say so in the report). Also: supplement trigger rate (B), wrong-
   supplement count (supplement made a correct answer wrong), extra latency p50/p95 and extra Gemini/search
   spend per question.
4. Pi pure-audio: only a 30 to 40 question slice of the same test half, arms A and B, to confirm the latency
   cost under real audio (T4). Never the whole set through live audio.

## Success criteria (written now, before any result)
- B is not worse than C by more than 2 points accuracy (paired CI includes 0) and does not lower unanswerable F1.
- Supplement trigger rate on in-scope non-stale questions below 5%; every trigger inspected by hand.
- Added time to first real answer p95 below 1.5 s offline (the verdict is ~0.8 s, speak-first hides the rest).
- C is not worse than A (fixes must not regress).

## Spend (needs owner approval before running)
Estimate before the run: calls = arms (3) x test half (~100) x ~3 Gemini calls + B verdicts, about $1 to $3 at
Flash-Lite prices, plus web search; the live-spend cap in memory ($0.50 total) is lower than this, so the
owner must raise or approve a specific cap first. Qwen scoring is free.

## Non-Goals
No tuning on the test half; no model or prompt changes inside this ticket; no deployment; `SUFFICIENCY`
stays False in code; FRAMES and MiMoTable follow after this.
