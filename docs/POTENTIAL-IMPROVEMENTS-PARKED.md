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
