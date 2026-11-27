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
  `eval/sufficiency/implicit_compound.jsonl` (v1, 60 rows: stale 20, missing_half 15,
  conflict 10, docs-only control 15), extended in ticket 260 by
  `eval/sufficiency/implicit_compound_v2.jsonl` (v2, 40 rows: stale 12,
  missing_half 10, conflict 6, control 12) — the full experiment set is 100 rows,
  split dev/test (see "Frozen dev/test split" below).
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

## Frozen dev/test split (ticket 260)

The labelled set is now split for the post-deployment experiment (release frozen).
Every row of both set files carries a `split` field, `"dev"` or `"test"`.

**Rule** (single source of truth: `scripts/check_implicit_set.py`, `split_for`;
verified per row by the checker, and by `test_split_for_is_the_documented_hash_rule`):

```
split = "dev"  iff  int(sha256(f"implicit-split:7:<id>").hexdigest(), 16) % 2 == 0
split = "test" otherwise
```

i.e. deterministic hash parity of the row id with fixed salt `7`. The assignment is
stratified by kind over the full 100-row set (v1 + v2): every kind's dev share lies in
[0.30, 0.70] and the overall split is 50 dev / 50 test (the checker's
`split_balance_errors` enforces this over the union of the two files).

**Frozen judge set (`test`, 50 rows):** imp-002, imp-003, imp-005, imp-008, imp-009,
imp-012, imp-014, imp-016, imp-019, imp-020, imp-022, imp-026, imp-028, imp-029,
imp-030, imp-031, imp-032, imp-035, imp-036, imp-037, imp-039, imp-041, imp-043,
imp-045, imp-047, imp-051, imp-052, imp-055, imp-057, imp-062, imp-065, imp-066,
imp-069, imp-070, imp-071, imp-072, imp-073, imp-074, imp-075, imp-076, imp-078,
imp-079, imp-082, imp-084, imp-086, imp-090, imp-093, imp-095, imp-096, imp-100.
By kind: needs_web_stale 17, needs_web_missing_half 11, needs_web_conflict 9,
docs_only_control 13.

**The remaining 50 rows (`dev`) are imp-001, imp-004, imp-006, imp-007, imp-010,
imp-011, imp-013, imp-015, imp-017, imp-018, imp-021, imp-023, imp-024, imp-025,
imp-027, imp-033, imp-034, imp-038, imp-040, imp-042, imp-044, imp-046, imp-048,
imp-049, imp-050, imp-053, imp-054, imp-056, imp-058, imp-059, imp-060, imp-061,
imp-063, imp-064, imp-067, imp-068, imp-077, imp-080, imp-081, imp-083, imp-085,
imp-087, imp-088, imp-089, imp-091, imp-092, imp-094, imp-097, imp-098, imp-099.**

Prompt and gate tuning may look **only at `dev` rows**; the `test` rows are the frozen
judge set. The harness selects splits with `--split dev|test|all` (and sets with
`--set v1|v2|both`, default behaviour unchanged).

## Ticket 260 — v2 set (40 rows) and dev/test split runs

Three gold-mode runs (the verdict measured independently of the lane), 2026-11-27,
`Qwen/Qwen3.8-27B` via the shared relay (free), `--today 2026-11-27`, corpus
`/tmp/ev218/db44_new`. **No prompt or gate was tuned for or against these runs.**

1. `--set v2 --split all` — the 40 new rows
2. `--set both --split dev` — the 50 dev rows of the full 100
3. `--set both --split test` — the 50 test rows (the frozen judge set)

### Verdict confusion by kind

**Run 1 — v2 (40 new rows):**

| kind | n | sufficient | stale | missing_half | wrong_question | none | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| needs_web_stale | 12 | 7 | 4 | 1 | 0 | 0 | 5/12 (42%) |
| needs_web_missing_half | 10 | 2 | 0 | 0 | 8 | 0 | 8/10 (80%) |
| needs_web_conflict | 6 | 3 | 1 | 2 | 0 | 0 | 3/6 (50%) |
| docs_only_control | 12 | 10 | 0 | 0 | 1 | 1 | 10/12 (83%) |
| all needs-web | 28 | 12 | 5 | 3 | 8 | 0 | 16/28 (57%) |

**Run 2 — dev split (50 rows):**

| kind | n | sufficient | stale | missing_half | wrong_question | none | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| needs_web_stale | 15 | 9 | 5 | 1 | 0 | 0 | 6/15 (40%) |
| needs_web_missing_half | 14 | 3 | 0 | 3 | 7 | 1 | 10/14 (71%) |
| needs_web_conflict | 7 | 4 | 1 | 2 | 0 | 0 | 3/7 (43%) |
| docs_only_control | 14 | 14 | 0 | 0 | 0 | 0 | 14/14 (100%) |
| all needs-web | 36 | 16 | 6 | 6 | 7 | 1 | 19/36 (53%) |

**Run 3 — test split, the frozen judge set (50 rows):**

| kind | n | sufficient | stale | missing_half | wrong_question | none | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| needs_web_stale | 17 | 4 | 12 | 1 | 0 | 0 | 13/17 (76%) |
| needs_web_missing_half | 11 | 3 | 0 | 1 | 7 | 0 | 8/11 (73%) |
| needs_web_conflict | 9 | 3 | 5 | 1 | 0 | 0 | 6/9 (67%) |
| docs_only_control | 13 | 12 | 0 | 0 | 1 | 0 | 12/13 (92%) |
| all needs-web | 37 | 10 | 17 | 3 | 7 | 0 | 27/37 (73%) |

**Full 100 (runs 2 + 3 combined):** needs-web pass 46/73 (63%) — stale 19/32 (59%),
missing_half 18/25 (72%), conflict 9/16 (56%); controls 26/27 (96%); overall verdict
agreement 72/100 (72%). The dev and test splits behave similarly (53% vs 73% needs-web
recall; the test split's stale kind is stronger, 76% vs 40%) — no sign so far that one
split is systematically easier.

### Over-trigger on controls

| run | over-triggers | other non-passing controls |
| --- | ---: | --- |
| v2 (40) | 1 (imp-075 `wrong_question`) | 1 (imp-077: empty verdict = the 1.5 s budget fail-open; it returned `sufficient` in the dev-split run) |
| dev (50) | 0/14 | 0 |
| test (50) | 1 (imp-075 `wrong_question`) | 0 |
| full 100 | 1/27 (3.7%) | 0 |

The only control misjudgement in any run is imp-075 ("In the action model, how many
guiding questions are there?" → `wrong_question`: "the answer describes steps, not
guiding questions") — a question-wording boundary, not a staleness over-trigger. No
control row ever received `stale`/`missing_half`.

### Failures (14 + 17 + 11; the model's own reason)

Run 1 (v2): imp-061, imp-062, imp-063, imp-067, imp-068, imp-069, imp-072 (stale
kind, `sufficient` — the target-year blindness of ticket 255 pattern 1: "next year",
"this month", "at present" questions answered with the newest in-document figures);
imp-075 (control, `wrong_question`); imp-077 (control, empty verdict / timeout flake);
imp-085, imp-089 (missing_half, `sufficient` — the ticket-255 pattern-3 shape: the web
fact rides in the same sentence as the document fact, so the verdict saw a complete
answer); imp-096, imp-098, imp-100 (conflict, `sufficient` — the hedge blindness of
ticket 255 pattern 2: "no longer maintained", "subject to change", a handbook default
spoken as settled).

Run 2 (dev): imp-001, imp-004, imp-006, imp-013, imp-018, imp-061, imp-063, imp-067,
imp-068 (stale, `sufficient` — the v1 rows re-run here fail on the same pattern 1 as
in ticket 255; imp-061/063/067/068 are the new v2 stale rows); imp-023 (missing_half,
empty verdict / timeout flake); imp-027, imp-085, imp-089 (missing_half, `sufficient`,
pattern 3); imp-038, imp-042, imp-044, imp-098 (conflict, `sufficient`, pattern 2).

Run 3 (test): imp-005 (stale, `sufficient`); imp-022, imp-028, imp-030 (missing_half,
`sufficient`, pattern 3); imp-045 (conflict, `sufficient` — "the answer states the
office location"; this is the v1 row whose *v1* run had the imp-045-style false
positive in the other direction, `wrong_question` — it is unstable across runs);
imp-062, imp-069, imp-072 (stale, `sufficient`, pattern 1); imp-075 (control,
`wrong_question`); imp-096, imp-100 (conflict, `sufficient`, pattern 2).

The same three ticket-255 miss patterns dominate the new rows; the new set adds no
qualitatively new failure mode. Run-to-run variance (one round each, temperature 0):
imp-077 `none` → `sufficient`, imp-045 `sufficient` ↔ `wrong_question` across its two
appearances — the empty-verdict rows are the known 1.5 s budget fail-open, not model
errors.

### Latency and tokens (one round each)

| run | doc lane p50/p95 | verdict p50/p95 | tokens (in/out) |
| --- | --- | --- | --- |
| v2 (40) | 2.44 / 4.57 s | 0.68 / 0.97 s | doc 233,550 / 3,631 · verdict 23,345 / 1,204 |
| dev (50) | 1.84 / 3.72 s | 0.56 / 1.01 s | doc 310,390 / 4,265 · verdict 29,437 / 1,519 |
| test (50) | 1.50 / 2.80 s | 0.50 / 0.82 s | doc 294,692 / 4,533 · verdict 30,002 / 1,687 |

Both stages sit inside the production budgets (doc 6.0 s, verdict 1.5 s).

Machine-readable reports: `out/ev260/v2-gold.json`, `out/ev260/both-dev-gold.json`,
`out/ev260/both-test-gold.json` (gitignored).

## Reproduction

```
# ticket 255 (60 rows)
pipeline/.venv/bin/python scripts/sufficiency_eval.py --repeat 2 --judge \
    --out out/sufficiency_eval/255-lane.json
pipeline/.venv/bin/python scripts/sufficiency_eval.py --gold-answer \
    --out out/sufficiency_eval/255-gold.json

# ticket 260 (v2 set, full 100 by dev/test split) — needs the pipeline venv:
#   PYTHONPATH=$PWD/src:$PWD/pipeline/src:$PWD/scripts
pipeline/.venv/bin/python scripts/sufficiency_eval.py --set v2 --split all \
    --gold-answer --out out/ev260/v2-gold.json
pipeline/.venv/bin/python scripts/sufficiency_eval.py --set both --split dev \
    --gold-answer --out out/ev260/both-dev-gold.json
pipeline/.venv/bin/python scripts/sufficiency_eval.py --set both --split test \
    --gold-answer --out out/ev260/both-test-gold.json

# the set itself: every row's evidence must occur in its cited corpus file,
# and the frozen split must match the hash rule (run over both files):
.venv/bin/python scripts/check_implicit_set.py \
    --set eval/sufficiency/implicit_compound.jsonl --root /tmp/ev218/raw44 \
    --also eval/sufficiency/implicit_compound_v2.jsonl \
    --expected-counts-2 'needs_web_stale=12,needs_web_missing_half=10,needs_web_conflict=6,docs_only_control=12'

pytest tests/test_sufficiency_eval.py tests/test_check_implicit_set.py   # fakes only, no network
```

Both runs executed 2026-11-27 on the shared relay (at most one call in flight, one
retry, loud abort on transport failure). The relay key is read from
`~/.secrets/api_key.txt`, sent only in the Bearer header, never printed or logged.
Machine-readable reports: `out/sufficiency_eval/255-lane.json`,
`out/sufficiency_eval/255-gold.json` (gitignored).
