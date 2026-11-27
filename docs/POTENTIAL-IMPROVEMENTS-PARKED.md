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
