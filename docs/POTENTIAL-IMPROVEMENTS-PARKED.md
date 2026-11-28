# Parked improvements (not scheduled), 2026-11-28

## P1: answerable questions the document lane does not accept
Source: lane runs on the 100-row sufficiency set (`out/sufficiency_eval/gemini-{dev,test}-lane.json`,
Qwen as the lane answerer, so Gemini's acceptance rate is unmeasured).

Lane outcomes by kind across both halves (accepted / none / check_failed): documents-only controls
20/3/4 (of 27), stale 10/17/5 (of 32), conflict 10/6/0 (of 16), missing half 0/24/1 (of 25).
Most non-accepted needs-web rows are the intended behaviour: the documents lack the current or
missing part, the lane declines, and the miss path (web) takes over. The sufficiency check only runs on
accepted document answers, so 40/100 accepted does not change production; it limits how much of the
measurement is end to end (the other 60 rows score the verdict on the gold document half).

The real weakness: 7 of 27 answerable controls (26%) were not accepted (3 `none`, 4 `check_failed`),
for example "How much RAM does the pico have?". Those turns fall to the miss path unnecessarily.
Not yet traced: which stage fails (retrieval, the answer step, or the checker).
Next step if picked up: a read-only trace of those 7 rows through the lane, then rerun the lane on
Gemini as the answerer to see whether the rate is a Qwen artefact.

## P2: time-robustness set for the date logic
All runs use one fixed "today" (2026-11-27). Not covered: other dates (Sep 30 / Oct 1 fiscal-year
boundary, Dec 31 / Jan 1), explicit-year questions that must stay `sufficient` on an old source ("What was
the 2024 rate?"), relative words beyond "now"/"next", spoken dates. Ticket 268 adds boundary unit tests;
a scored set is still missing. Write it from dev-side rows only.

## P3: conflict rows without a date cue
Confident undated facts (imp-036, 037, 039, 045, 100 and the test misses on Gemini: imp-100, imp-036) are
the weak kind on both Qwen and Gemini. Dev conflict is saturated, so tuning needs new dev-side rows
written without looking at test rows. Ticket 267 (a volatility check) was a negative result.

## P4: Gemini as the lane answerer in the eval
The harness only swaps the verdict model. Running the lane on Gemini would make the lane-accepted numbers
production-representative (paid, about the same call count again).

## P5 (reference): grounded Gemini web-search critique
Source: `origin/qwen/ingest` `62185ff`, `docs/WEB-SEARCH-CRITIQUE-2026-11-28.md` (proposal) and
`docs/evidence/WEB-SEARCH-CRITIQUE-2026-11-28.md` (evidence, plus `docs/evidence/web_search_audit_probes.py`).
Not read from this checkout's code: I read the two documents only and did not rerun its probes (the
document reports 137 offline tests passing and four mocked probes reproducing the defects).

Why it matters for sufficiency: the supplement sentence goes through the same grounded web lookup, so
its quality is bounded by these web-layer findings. Summary of the critique:
- P0: a grounded non-answer (the hedge) returns as non-null text and ends the fallback chain (a mocked run
  made two Gemini searches, two directness checks and zero Tavily requests); one lookup result is shared
  across `web_answer`, `web_research` and Wikipedia for the whole turn, so "search harder" cannot go deeper.
- P1: the directness judge sees only the generated answer, not source passages, and its word-overlap guard
  accepts a negation inversion; rechecks can reuse the cached evidence the user challenged (10 min / 1 h
  TTLs); research snippets are clipped to 400 characters (2,000 total) with no raw content; Gemini gets 3 s
  per attempt.
- Proposed (unmeasured) design: a structured outcome (answered / insufficient / conflicting / timeout /
  failure), a bounded Gemini-only recovery that changes the query and reads up to two public URLs
  (Search plus URL Context, trial bound 15 s and two extra grounded calls), platform/edition/as-of scoping
  for "latest" questions, and an archived-evidence plus repeated-live-run evaluation.
- Its open items: Pi provider traces for the Nov 27 incidents, and an owner-approved live Gemini subset.

Relation to our work: it argues for a structured "insufficient evidence" outcome, which the sufficiency
supplement would also need (today `HEDGE_TAIL` in `sufficiency.py` is string-matched). Parked with the agy
critique fixes T01 to T10; the owner's order (sufficiency, then MMLongBench-Doc) is unchanged.

### P5 check, 2026-11-28 (offline, read-only; no paid calls)
- Ran the critique's four mocked probes against this branch's web layer (probe file read from `origin/qwen/ingest`,
  run with the repo venv, nothing written): all four reproduce here (`nonanswer_ends_fallback` 4 Google requests /
  0 Tavily, `platform_cache_collision` same_question true, `negation_inversion` supported true,
  `snippet_cutoff` winner dropped). So the web-layer defects are real on exp/sufficiency too.
- Effect on the supplement (`agent_server.py` `_maybe_supplement`, ~line 976):
  - Hedge / non-answer: safe. The supplement drops any reply containing `HEDGE_TAIL` and says nothing, so the
    P0 chain-termination costs a missed supplement, not a wrong one. Tavily is not configured anyway, so
    grounded Gemini is the only backend.
  - Negation / directness: a real risk. A reply that passes the directness check is spoken as "The web has a
    newer figure: <text>" (stale) with nothing verifying it is a figure or newer; the lead asserts it. Not
    measured on real web replies.
  - Rechecks: the supplement is one non-retried lookup, so cached-evidence reuse matters only if the user
    repeats the question inside the 10-minute TTL.
- Not done (needs the owner's approval, paid): run the 36 dev needs-web `web_question`s through the real
  grounded lookup and hand-read what the supplement would say (count wrong/unsupported "newer figure"
  claims). That is the measurement that would show whether the critique's recovery design is worth building
  before the sufficiency flag is considered.

### P5 measured, 2026-11-28: what the supplement would say (34 dev web questions, real grounded lookup)
Script `scripts/supplement_web_check.py`, report `out/sufficiency_eval/supplement-web-check.json` (not committed).
Cost: the ledger shows about 26 searches (worst case $0.014 each, about $0.36; the free monthly quota would
make it lower).
- 4 of 34 replies were the hedge (imp-010, 040, 063, 064): spoken as nothing, harmless.
- Of the other 30, my hand read finds about 6 that the supplement would speak wrongly or misleadingly
  (facts not independently verified; this is a reading of the reply against the question):
  - not a figure, but led by "The web has a newer figure:": imp-011 ("October 2026 growth has not yet been
    released"), imp-015 (fall break "has already passed");
  - wrong scope: imp-004 (asked about the US GS-15 pay, answered with the Philippine salary grade), imp-013
    (a Cebu university schedule for a question with no place);
  - the web answer is itself out of date: imp-001 and imp-061 give the FY2026 per diem ("fiscal year ending
    September 30, 2026") on 2026-11-27, when FY2027 rates are the current ones.
- The rest read as plausible direct answers with named sources.
Reading: the supplement's quality is bounded by the web layer, as the critique says. The cheapest
mitigations that do not need the recovery design: make the lead sentence depend on a check that the reply
states a figure (not "not yet released"/"has passed"), drop replies that name a period older than the
target, and keep the web question scoped to what the user asked. None implemented.

### Ticket 268 verification (orchestrator, 2026-11-28)
Qwen committed `ad717b8`. Re-run: pipeline suite 1695 passed, 1 xfailed, only the known `test_docs_wiring`
failure; ruff clean; dev gold eval on Qwen unchanged (needs-web 35/36, stale 15/15, conflict 7/7, controls
0/14, 11 fast-path stale rows); `pi_clock_check.sh` passes `bash -n`, reports by default and calls
`sudo date -s` only under `--fix`. One naive `date.today()` is left in `src/voice_agent/agent/subagent.py:474`
(outside the sufficiency path, not touched).

### Ticket 270 held-out check on the test half, 2026-11-28 (one run, no tuning afterwards)
35 test web questions through the real grounded lookup (`scripts/supplement_web_check.py ... gemini-test-gold.json`),
then the 270 gates (`supplement_ok`, now=2026-11-27 Manila). Cost: about 21 searches (worst case $0.29;
dev run about $0.36, so the live-spend total is at or past the $0.50 cap in the worst-case accounting).
- 4 hedges (silent). 31 real replies: 25 kept, 6 dropped.
- Dropped, my hand read: imp-002, imp-072 (FY2026 per diem "through September 30, 2026": correctly dropped),
  imp-009 (rates "vary", no figure: correct), imp-043 (describes the program, never names the teacher: correct),
  imp-014, imp-016 (the 2026-27 year that already began, asked about next year: reasonable).
  One wrong drop: imp-096 ("is the ATSDR primer still maintained?", answer "No, no longer maintained"): a good
  yes/no reply dropped by the figure gate. The figure gate wrongly treats a yes/no or a name as "no figure" when
  the verdict is `stale`; better to change the lead than to drop.
- Kept but misleading: imp-012 (asked when classes start next fall, reply says the current year began June 8,
  2026), imp-037 (a university question answered from the Philippine DepEd calendar; fixture has no doc answer
  so the scope check is skipped). Uncertain, not verified: imp-039 (WOTUS "currently follows the 2023 rule").
- Rough rate by my reading (facts not independently verified): without gates about 8 of 31 replies misleading
  (about 26%); with gates about 2 to 3 of 25 spoken (about 8 to 12%).
Follow-ups (not done): yes/no and named answers under the figure gate; a "reply answers a different period than
asked" check (imp-012); the scope check cannot see place guesses inside a country or in rows with no doc answer.

## P6: `key_absent` rejects correct answers on spelling differences (found 2026-11-28, MMLongBench-Doc dev, ticket 274)
`_key_tokens` (`ask_documents.py:690`) takes capitalised mid-question words as key tokens and `_key_absent` rejects the
answer if one is literally absent from the cited file's passages. Spoken or benchmark spellings differ from the
document, so a verbatim-correct answer is refused. Reproduced with `_key_tokens` on three dev rows whose gold was in the
retrieved passage: "FY2021" (the 10-K says "fiscal 2021") for Costco q1025; "FY2023" and "Bestbuy" (the file says "Best
Buy") for q1083; "Erstwhile", "Hotels", "Limited", "Rupee" (the table uses a symbol and digits) for ITC q0939. `key_absent` was
8 of 13 `check_failed` rows. The check is in the frozen release too. Not a crash; parked. Candidate fixes to evaluate
(not tuned on the test half): fold "FY<yyyy>"/"fiscal <yyyy>", drop the space ("bestbuy" matches "best buy"), skip tokens
that name a currency or unit. Measure on the 100-row sufficiency set and the MMLongBench-Doc dev rows before and after;
the check exists to stop wrong-document answers, so count the wrong-document rejects it stops today (controls).

## MMLongBench-Doc dev diagnosis, hand-checked (ticket 274)
Qwen's "not self-contained" judge put 43 of the 62 no-answer rows in R0 and said 60 of 82 dev rows are not
self-contained. A hand read (`eval/mmlongdoc/hand_r0_dev.json`) rates 27 of those 43 not self-contained and 16 self-
contained (company or topic named: Nike, Netflix, NTU, Godfrey Phillips, ProgramFC, HOVER ...). Those 16 are real lane
or retrieval misses, mostly values that must be computed from statements (ratios, percentage change) plus 2 rows in an
empty slide deck. The 17 accepted rows the judge marked not self-contained were not hand-rated. So the shared-corpus
ceiling is above the 27% implied, not the 27% of 82 Qwen implied: at least 27 of 82 rows (33%) cannot be answered from a
shared corpus, possibly a few more among accepted rows. Two slide-deck PDFs yield zero chunks (reader gap, 5 dev rows).
