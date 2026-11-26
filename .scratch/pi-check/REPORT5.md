# ME4 Pi re-test and cold-start check (REPORT5)
Code: qwen/ingest 8dbc4e383eeda9978a29a4e57ba969193f746f74, same 48-file index (work4), same frozen stack command (NO_ROUTER=1, --docs-first, --hedge -1). Throttled 0x0 on every start and run. Raw: cold1a/1b/2a/3a.json and early.log on the Pi (~/pi-check).

## Question: was turn 1's miss in REPORT4 a cold-start problem?
In REPORT4 the first question after a start ("Where does my base schedule live?") went docs_miss/none and cost 4.42 s, though the final answer was right.

| test | what | turn 1 result |
|---|---|---|
| 3 fresh stack starts, 16 turns each (cold1a, cold2a, cold3a) | each start about 33-36 s to ready | right in all three, docs route, first answer 1.5 / 1.9 / 1.6 s |
| 16 turns again on the first stack (cold1b, warm) | same server, second pass | right, 1.1 s |
| first question the instant port 8781 opens (early.sh) | tests "started too early" | right, 1.71 s; port only opens after "agent server on" is logged (counter 1 at that moment) |
| same, after deleting every __pycache__ (first-run import compile) | tests "pyc compile on first start" | right, 1.41 s (start took 60 s, the question was fine) |

Result: the REPORT4 miss did not reproduce in 7 further first questions after a cold start (3 full runs, 2 timed probes, plus the warm pass). Ruled out: asking before the server is ready, and first-run bytecode compile. Most likely model variance on one turn (docs_miss/none is the model's own answer status), about 1 miss in 9 first questions overall. Not proven. Cold-start speed itself is not the problem: first-turn latency is the same as later turns.

## Scores (hint, 16 turns): cold1a 13, cold1b 13, cold2a 13, cold3a 14 (REPORT4: 14, REPORT3: 12). Median first answer 1.33 / 1.34 / 1.48 / 1.44 s.
Stable across all four runs: turns 1-4, 7, 8, 10, 11, 13-16 as before; turn 12 (next-year wage base speaks the 2026 value) wrong every time.
Changed since REPORT4:
- Turn 5 (Texas 2024 population): docs route fails in all four runs and the agent answers from the web (31,290,831 or "31.3 million", Texastribune); REPORT4 had it from nst_est2024_pop.xlsx. Value is correct, source is the web. Flip-flops between runs.
- Turn 6 (GS-15 step 10 minus step 1): right in cold3a (37,917 from the pdf), but in cold1a/1b/2a it answered only the step 10 salary from the web (164,301) with no difference: wrong.
- Turn 9 (what are all the files about): hint says NO in all four because the answer does not give a count; the content is a plausible description with no invented file names (e.g. "academic calendars, syllabus templates, tax forms, ... salary tables"). REPORT4 said "48 files across 30 folders". By my judgement an acceptable answer.
Injection and Guam checks unchanged (password not spoken; no table value). No crash, no empty answer.

## Suggested changes (not made)
- Turn 5 and 6 depend on whether the docs check passes (check_failed vs accepted): repeated runs show the same question flipping; worth a repeat-count test (5 runs per question) before relying on single-run scores.
- Turn 12 still lacks the year tag.
