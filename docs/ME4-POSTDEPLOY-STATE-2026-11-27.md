# ME4 post-deployment state (2026-11-27, written right after the freeze)

Read `docs/ME4-RELEASE-MANIFEST-2026-11-27.md` first for the frozen release. This file says what exists beyond it, where it lives, what is still running, and the order of work. Everything here is post-deployment and must not change the frozen code (`1421c7c`, tag `me4-rc-2026-11-27`, branch `release/me4-2026-11-27`).

## Branches and worktrees

| Branch (origin) | Worktree | What it is | State |
|---|---|---|---|
| `release/me4-2026-11-27` | none | the frozen release plus manifest and Pi results | frozen; crash-class fixes only, re-tag |
| `qwen/ingest` | `~/AI-231-ME4-qwen` | integration branch: the frozen code plus docs/results | docs and Pi results only after the freeze |
| `exp/sufficiency` | `~/AI-231-ME4-w250` | frozen code + ticket 250 (sufficiency check, `SUFFICIENCY=False`) + 249 (60-question set) + 255 (offline eval harness and baseline) + agy's review/plan (merged from `exp/sufficiency-agy`) | pushed; pipeline suite 1600 passed, 10 xfailed (the documented bugs), 1 known failure |
| `exp/sufficiency-agy` | `~/AI-231-ME4-w256` | agy ticket 256: review, experiment plan, xfail tests | snapshot pushed (`c086740`); the agy run may still add to it |
| `crit/frozen-2026-11-27` | `~/AI-231-ME4-w257` | agy (Argon, `gemini-4-argon-high`) ticket 257: adversarial critique of the frozen release | RUNNING at the time of writing (60 min limit); output lands uncommitted in `docs/FROZEN-ADVERSARIAL-CRITIQUE-2026-11-27.md` and `.scratch/t/257.report.md` of that worktree |
| `w/248` | `~/AI-231-ME4-w248` | model research `docs/MODEL-RESEARCH-2026-11-27.md` | pushed |
| `w/249` | `~/AI-231-ME4-w249` | implicit-compound question set | merged into `exp/sufficiency` |

To pick up the critique when agy finishes: look in `~/AI-231-ME4-w257`, read the two files, then commit them with explicit paths (`git add docs/FROZEN-ADVERSARIAL-CRITIQUE-2026-11-27.md .scratch/t/257.report.md`), push `crit/frozen-2026-11-27`, and cherry the doc onto `qwen/ingest`. agy never commits (its allow-list forbids it). Same for ticket 256's final state.

## Sufficiency check: results so far (all offline, free, Qwen3.8-27B as answer and verdict model)

Baseline from `scripts/sufficiency_eval.py` on the 60 questions (`docs/SUFFICIENCY-EXPERIMENT-RESULTS-2026-11-27.md`, prompt NOT tuned):
- Over-trigger on the 15 docs-only controls: 0 of 15 (the one non-pass was a 1.5 s timeout that failed open).
- Recall on the 45 needs-web rows: 29/45 (64%) in lane mode; by kind: stale 15/20, missing_half 11/15, conflict 3/10.
- Verdict latency 0.55 s median, 1.03 s p95 (budget 1.5 s); about 263 prompt tokens per call.
- The document lane itself accepted only 22 of 60 of these implicit questions with Qwen (32 none, 6 check_failed): the lane, not the verdict, limits the end-to-end result.
- Misses cluster in three patterns: "next year" questions (the verdict compares the source year with today, not the question's target period), documents that hedge themselves ("proposed", "as of September"), and gold answers that carry the web fact in the same sentence (a data-marker artefact).

agy review (`docs/SUFFICIENCY-REVIEW-2026-11-27.md`, 4 R1 + 6 R2). I verified R1-1 myself:
- **R1-1** production `gemini_json_request` (ask_documents.py ~1215) forces `response_json_schema=REPLY_SCHEMA`, so a verdict reply cannot parse and `verdict()` returns None in production. The Qwen baseline bypassed this with its own request, so it does not show it. Fix first: give the verdict its own request function with its own schema (the 245 directness check already makes its own Gemini call).
- R1-2 `ask_documents(status="web")` double lookup and a spoken "(already looked up this turn)"; R1-3 `wrong_question` pays for a web lookup then discards it and poisons `turn_source_sentence`; R1-4 a WS disconnect or cancel leaks the verdict task and stalls the lock.
- R2: first audio is delayed by the SentenceFilter/space/40-word cap; `FY` is stripped before `source_earlier`; the gate under-triggers missing_half; no `offline()` guard; late supplement dead air; private document text can leak into the public `web_question`.
- 9 of its tests are `xfail(strict=True)` documenting the bugs, so the suite stays green and flips loudly when one is fixed.
- Plan (`docs/SUFFICIENCY-EXPERIMENT-PLAN-2026-11-27.md`): do not run live or ship-gate experiments until R1-1..R1-4 and R2-1..R2-6 are fixed; dev/frozen 60/40 split by document family (add +40 rows, 10 wrong_question, more controls with temporal-looking words); thresholds H1-H4 (recall >= 80% stale, >= 70% missing_half/conflict, over-trigger <= 10%, first-audio unchanged, <= 0.35 cheap calls and <= 0.15 searches per turn).

## Order of work after the test (owner's priorities first)

1. Read the Pi log for run 6/7 (`pipeline/out/agent_server/*.jsonl`) to find why Ethiopia electricity and similar rows go documents-miss then web on Gemini but are accepted offline.
2. Collect agy's frozen-release critique (ticket 257), merge its top-10 list with the queue below, and rank by measured headroom.
3. Sufficiency: fix R1-1..R1-4 and R2 items on `exp/sufficiency` (Qwen), extend the set 60 -> 100 with a frozen split (Qwen), re-run the baseline, then the full flow with a real Gemini subset (owner runs paid scripts with `!`).
4. Doc+web: widen the 247 gate or use the sufficiency verdict to trigger the split; a miss that mentions unreadable files; honest-miss wording that names the documents.
5. Wall-clock budget (ask_documents ~997 `budget_s - offset`), today's date and the file card into `ask_documents`, the prompt checklist (entity/row/column/unit/period), one bounded repair re-ask. Measure on a real-model harness (Qwen answers + judge), not the oracle gates.
6. Ingest hardening: zip staging bounds, dbindex lock, `file_has` via FTS, OCR on a zip, a corpus-derived ASR vocabulary.
7. Deep research behind a flag (`origin/feat/deep-research-skill-harness`).

## Tools and rules to remember

- Offline gate: `/tmp/zz235/gate.sh <worktree> <tag>` and row-level compare against `/tmp/ev218/base*.json` (scratch, not in the repo; rebuild from `scripts/me4_ask_eval.py` and `/tmp/ev218/eval_g2.py` if /tmp is gone). Fresh ingest for any reader change: `python -m voice_agent.rag.ingest <folder> --out <db>` with the root venv; bump `POSTPROC_VERSION` / `LEGACY_PARSE_VERSION` when reader output changes.
- Qwen: `~/.local/bin/qwen-dispatch <NN> <brief> <worktree> file,terminal,todo <timeout>` started with `setsid nohup ... > file < /dev/null &`; at most 2 runs; explicit `git add` paths. agy: `models/agy/agy-dispatch.sh <worktree> <ticket> <NN> [model]`; models listed by `agy models` (hydrogen, neon, carbon, argon tiers); allow-list has no python/curl/zip/commit.
- Judge: `scripts/judge_eval.py` (Qwen logprobs). Keyring and secrets: the owner runs anything that reads a key with `!`; OpenRouter key is in the keyring under service `openrouter` (`secret-tool lookup service openrouter`), parked bake-off script was `scratchpad/or_bakeoff.py` (not in the repo).
- Stand-in corpus for Pi tests: private HF dataset `martinnavs/me4-pi-package` (`standin_c2.zip`, questions, manifest); builders and random-fixture generators were in `/tmp/standin/` (`build.py`, `gen_make.py`, `gen_check.py`, `gen_tree.py`; need xlwt and reportlab in a venv); they are not in the repo.
