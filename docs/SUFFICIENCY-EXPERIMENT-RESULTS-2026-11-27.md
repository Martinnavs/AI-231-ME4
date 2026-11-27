# Sufficiency check — offline baseline results — 2026-11-27 (ticket 255)

Post-deployment offline evaluation of the speak-first sufficiency check (ticket 250,
`pipeline/src/voice_pipeline/sufficiency.py`, flag `SUFFICIENCY` off in the release).
**Baseline only: the prompt was not tuned to this question set; changes are proposed
below, not made.** The release is frozen (tag `me4-rc-2026-11-27`); nothing under
`pipeline/` changed in this ticket — deliverables are `scripts/sufficiency_eval.py`,
`tests/test_sufficiency_eval.py`, and this doc.

Setup:

- Harness: `scripts/sufficiency_eval.py` — corpus-agnostic (flags `--db`, `--jsonl`,
  `--today`, `--model`, `--base-url`, `--gold-answer`, `--repeat`, `--judge`,
  `--verdict-timeout`, `--out`); at most 1 relay call in flight; retries once; aborts
  loudly on a transport failure. 24 fake-transport tests (no network) in
  `tests/test_sufficiency_eval.py`.
- What runs per row: (1) the frozen document lane — `ask_documents` over
  `DbIndex.file_context` retrieval — and (2) the frozen `sufficiency.verdict`, both on
  the same shared cheap JSON model, exactly as in production.
- Model: `Qwen/Qwen3.8-27B` via the shared relay (temperature 0, thinking off,
  JSON-only response format) for both stages. `--today 2026-11-27` (fixed).
- Corpus: `/tmp/ev218/db44_new` (44 files) + labelled set
  `eval/sufficiency/implicit_compound.jsonl` (60 rows: stale 20, missing_half 15,
  conflict 10, docs-only control 15).
- Two modes: **lane** (the verdict sees the lane's spoken answer; a row the lane does
  not accept is scored on the document half of the gold ideal answer instead, so the
  verdict is measurable independently of lane failures) and **gold**
  (`--gold-answer`: the verdict sees the document half of the gold answer on every
  row). The document half strips the web-hedge sentences of the gold answer ("…verify
  live." / "(web)"), so the verdict is never handed the web fact.
- Production budgets the verdict must fit (frozen): verdict await `VERDICT_TIMEOUT_S =
  1.5 s` (speak-first: the verdict task starts before speaking and is awaited 1.5 s);
  doc turn budget `DOCS_TURN_BUDGET_S = 6.0 s`.

## Headline numbers (lane mode, the mode closest to production)

Table 1 — verdict confusion by kind (needs-web rows must come back triggered; controls
must come back sufficient):

| kind | n | sufficient | stale | missing_half | wrong_question | none | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| needs_web_stale | 20 | 5 | 13 | 2 | 0 | 0 | 15/20 (75%) |
| needs_web_missing_half | 15 | 4 | 0 | 4 | 7 | 0 | 11/15 (73%) |
| needs_web_conflict | 10 | 7 | 2 | 0 | 1 | 0 | 3/10 (30%) |
| docs_only_control | 15 | 14 | 0 | 0 | 0 | 1 | 14/15 (93%) |
| all needs-web | 45 | 16 | 15 | 6 | 8 | 0 | 29/45 (64%) |

Table 2 — over-trigger rate on controls (the main risk):

| mode | true over-triggers | other non-passing controls |
| --- | ---: | --- |
| lane | 0/15 | 1 (imp-058: verdict call hit the 1.5 s production budget → fail-open `none`, not a misjudgement; the repeat round returned `sufficient`) |
| gold | 0/15 | 0 |

**No control row ever received a triggered verdict in either mode.** The single
non-passing control was a timing flake, not a model error.

Table 3 — gold mode (the verdict measured independently of the lane):

| kind | n | sufficient | stale | missing_half | wrong_question | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| needs_web_stale | 20 | 7 | 12 | 1 | 0 | 13/20 (65%) |
| needs_web_missing_half | 15 | 4 | 0 | 4 | 7 | 11/15 (73%) |
| needs_web_conflict | 10 | 6 | 3 | 1 | 0 | 4/10 (40%) |
| docs_only_control | 15 | 15 | 0 | 0 | 0 | 15/15 (100%) |
| all needs-web | 45 | 17 | 15 | 6 | 7 | 28/45 (62%) |

Table 4 — stability and agreement (lane mode, `--repeat 2`):

| metric | value |
| --- | --- |
| rows with identical verdict across the 2 rounds | 58/60 |
| flips | imp-012 `missing_half` → `stale` (both triggered; no scoring effect); imp-058 `none` → `sufficient` (the 1.5 s timeout flake) |
| verdict agreement with the label (lane mode) | 43/60 (72%) |
| verdict agreement with the label (gold mode) | 43/60 (72%) |
| cross-mode verdict agreement (same row, lane vs gold input) | 53/60 |

Table 5 — latency vs the production budgets (p50/p95, all rounds):

| stage | p50 | p95 | production budget |
| --- | ---: | ---: | ---: |
| document lane (lane mode) | 1.30 s | 3.18 s | 6.0 s |
| document lane (gold mode) | 1.12 s | 2.49 s | 6.0 s |
| verdict (lane mode, 120 calls) | 0.55 s | 1.03 s | 1.5 s |
| verdict (gold mode, 60 calls) | 0.50 s | 0.74 s | 1.5 s |

Over-budget verdict calls: **1 of 120** (imp-058 at 1.502 s, +2 ms over → fail-open).
The verdict fits the speak-first budget with ~2× headroom at p95.

Table 6 — token cost per mode (one round = 60 rows):

| stage | prompt | completion |
| --- | ---: | ---: |
| document lane | 371,532 | 5,128 |
| verdict | 31,442 | 1,903 |

≈263 prompt tokens per verdict call — a cheap check.

Table 7 — judge directness of the spoken lane answers (`--judge`, 22 accepted rows):

| label | n |
| --- | ---: |
| A — answers the question | 20 |
| B — describes only (imp-012, conf 0.63) | 1 |
| C — not found (imp-045, conf 0.54) | 1 |

Mean judge confidence 0.90.

Context: the lane itself accepted only **22/60** rows (37%); 32 came back `none` and 6
`check_failed`. Lane mode therefore scored 38 rows on the gold document half — which is
why gold mode exists, and why the two modes produce near-identical verdict numbers
(43/60 in both; 53/60 cross-mode).

## Failures (lane mode, 17 rows; the model's own reason)

| id | kind | verdict | doc/source | model reason |
| --- | --- | --- | --- | --- |
| imp-004 | needs_web_stale | sufficient | accepted/lane | the answer states the specific salary amount |
| imp-005 | needs_web_stale | sufficient | check_failed/gold | the answer states the 2026 wage base |
| imp-006 | needs_web_stale | sufficient | check_failed/gold | the answer states the 2026 cap |
| imp-013 | needs_web_stale | sufficient | none/gold | the answer states the requested date |
| imp-018 | needs_web_stale | sufficient | check_failed/gold | the answer states the 2026 FUTA wage base |
| imp-022 | needs_web_missing_half | sufficient | none/gold | the answer states the time zone |
| imp-027 | needs_web_missing_half | sufficient | none/gold | The answer directly states the headquarters location. |
| imp-028 | needs_web_missing_half | sufficient | none/gold | the answer states the original publisher |
| imp-030 | needs_web_missing_half | sufficient | none/gold | the answer states the distance from Chicago |
| imp-037 | needs_web_conflict | sufficient | accepted/lane | the answer directly addresses the question |
| imp-038 | needs_web_conflict | sufficient | none/gold | the answer states the finals dates |
| imp-039 | needs_web_conflict | sufficient | accepted/lane | the answer states the definition as asked |
| imp-040 | needs_web_conflict | sufficient | accepted/lane | the answer states the issued laptop type |
| imp-042 | needs_web_conflict | sufficient | none/gold | the answer states the top coping method |
| imp-043 | needs_web_conflict | sufficient | none/gold | the answer states the instructors as asked |
| imp-044 | needs_web_conflict | sufficient | none/gold | the answer states growth continuing through September 2026 |
| imp-058 | docs_only_control | none | accepted/lane | (verdict call timed out at the 1.5 s budget; fail-open) |

The 16 model misses cluster into three patterns:

1. **"Next year" questions** (5 rows: imp-004, 005, 006, 013, 018): the verdict
   compares the source year against *today*, not the question's target period. The
   answers quoted current-year (2026) figures for "next January" / "next year"
   questions and were called sufficient. imp-004 was also blind: its file card has no
   date (source_year None).
2. **Hedged / conflicting content** (7 rows: imp-037, 038, 039, 040, 042, 043, 044):
   the model treated document-stated facts as complete even when the document itself
   hedges ("proposed", "tentative", "as of September 2026", a 2020 training module
   presented as current). The conflict kind is the weakest: recall 30% (lane) / 40%
   (gold).
3. **Single-sentence gold answers with the web fact inline** (4 rows: imp-022, 027,
   028, 030): the harness keeps a sentence when *every* sentence is web-marked (a row
   must never be scored on nothing); in this data shape the web fact rides in the same
   sentence as the document fact, so the verdict saw a complete answer and was correct
   to call it sufficient. This is a data-marker convention, not a model error. All 4
   two-sentence missing_half rows were caught.

## 5 rows where the check helped (would have supplemented live in production)

| id | kind | what was spoken / judged | verdict (web_question) |
| --- | --- | --- | --- |
| imp-001 | stale | "$209" (FY2026 per diem) for "right now" | stale — "What is the current hotel per diem rate in Boston?" (reason: source is from 2025, today is 2026) |
| imp-041 | stale | 2024 state-population ranking for "right now" | stale — "Which state has the largest population in 2026?" (source is from 2024, not current) |
| imp-036 | conflict | "Kathleen Vanderveen" (2024 list, interim) for a present-time question | stale — "Who is the current Title IX coordinator at GVSU?" (question asks for the present coordinator) |
| imp-021 | missing_half | "the last decennial count was 2020" for "when is the next census?" | missing_half — "When is the next US decennial census?" (answer only states the last census, not the next) |
| imp-035 | missing_half | "Raspberry Pi is a UK company" for "where is it based?" | missing_half — "Where is Raspberry Pi Ltd. based?" (names the company but not its location) |

All five had non-empty `web_question`s, so the frozen flow would have appended the live
supplement sentence ("The web has a newer figure: …" / "From the web: …").

## 5 rows where the check hurt or missed

| id | kind | what happened |
| --- | --- | --- |
| imp-004 | stale | "$126,384" (the 2026 table) spoken for "next January"; no trigger — the file has no card date (source_year None) and the check never compares the question's target year |
| imp-013 | stale | the 2026-27 fall start spoken for "next year"; no trigger (same target-year blindness) |
| imp-039 | conflict | a 2020 WOTUS definition spoken as settled; no trigger — the card date (1983) is outside the 20xx regex, so source_year was None |
| imp-044 | conflict | "Is the UK economy still growing?" answered with data through September 2026 (two months old); no trigger — no notion of data-currency without a source year |
| imp-045 | conflict | false positive: the correct room number "AIRC 3005" logged as `wrong_question` ("a course code, not a location"). Production impact is log-only (no retraction); the judge was also unsure (C, conf 0.54) |

## Actionable subset (what production would actually do)

`wrong_question` is logged, not spoken, in the frozen flow. Stale/missing_half with a
web question is the actionable subset:

| mode | stale kind | missing_half kind | conflict kind | all needs-web |
| --- | ---: | ---: | ---: | ---: |
| lane | 15/20 | 4/15 | 2/10 | 21/45 |
| gold | 13/20 | 4/15 | 4/10 | 21/45 |

## Structural limitation: source-year availability

The verdict is handed the file card's `date_max` year when available (else None) —
the same inputs the frozen production path uses. In this corpus the source year was
known for only **19/60 rows**; 41 rows had no card date, and some known values are
text-derived artefacts: 2027 on 3 rows (e.g. imp-015: "the source is from a later
year"), 2029 on 3 rows (imp-017: "the source is from 2029, not the current fiscal
year" — a future date), 2008 on 2 rows (imp-034: "the source is from 2008, far
outdated"). The check is largely blind to source age here; pattern-1 and part of
pattern-2 misses are direct consequences.

## Proposed changes (for a follow-up ticket — none made here)

1. **Target-period awareness.** Teach the verdict to compare the question's target
   period with the answer's stated year ("next January" + a 2026 figure → stale).
   Fixes the 5 pattern-1 misses without touching the over-trigger rate.
2. **Hedge cues.** Treat document-side hedges ("proposed", "tentative", "as of
   <date>", "interim") as stale/missing_half triggers. The conflict kind is the
   weakest (30%/40% recall).
3. **Better source dates.** Propagate ingest-time publication/mtime into `file_cards`
   (19/60 availability is too low) and clamp or drop future `date_max` artefacts.
4. **missing_half vs wrong_question boundary.** 7 missing_half rows came back
   `wrong_question` (log-only in production). Prefer `missing_half` when the topic
   matches but a fact is absent — while guarding against imp-045-style false
   positives (a room number read as a course code).
5. **Timeout.** 1/120 verdict calls (0.8%) silently fail open at the 1.5 s budget;
   no over-triggers were observed, so speak-first's safety is intact. If fewer silent
   skips are wanted, raising `VERDICT_TIMEOUT_S` to 2.0 s keeps ~2× headroom over the
   observed p95 (1.03 s) — the verdict task already starts before speaking.

## Reproduction

```
pipeline/.venv/bin/python scripts/sufficiency_eval.py --repeat 2 --judge \
    --out out/sufficiency_eval/255-lane.json
pipeline/.venv/bin/python scripts/sufficiency_eval.py --gold-answer \
    --out out/sufficiency_eval/255-gold.json
pytest tests/test_sufficiency_eval.py   # 24 tests, fakes only, no network
```

Both runs executed 2026-11-27 on the shared relay (at most one call in flight, one
retry, loud abort on transport failure). The relay key is read from
`~/.secrets/api_key.txt`, sent only in the Bearer header, never printed or logged.
Machine-readable reports: `out/sufficiency_eval/255-lane.json`,
`out/sufficiency_eval/255-gold.json` (gitignored).
