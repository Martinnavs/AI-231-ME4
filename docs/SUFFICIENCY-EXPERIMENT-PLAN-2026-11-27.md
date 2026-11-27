# Sufficiency Check Experiment Plan (2026-11-27)

Decision plan to determine whether to enable `SUFFICIENCY=True` on ME4 after `me4-rc-2026-11-27`. **Pre-condition:** do not run live or ship-gate experiments until R1-1..R1-4 and R2-1..R2-6 in [SUFFICIENCY-REVIEW-2026-11-27.md](SUFFICIENCY-REVIEW-2026-11-27.md) are fixed on an experimental branch.

## 1. Hypotheses

- **H1 (Verdict accuracy):** On accepted document answers, the cheap verdict model + fixed gate recovers `>= 80%` of `stale` and `>= 70%` of `missing_half`/`conflict` cases without exceeding `<= 10%` false triggers on `docs_only_control` rows.
- **H2 (End-to-end spoken quality):** Adding the web supplement improves completeness/correctness (via `scripts/judge_eval.py`) on implicit-compound questions while keeping `directness` and `faithfulness` high and adding zero retractions.
- **H3 (Zero first-audio regression):** With `flush: true` (R2-1) and bounded timeouts, pure-audio `speech_end_to_first_audio_ms` p50/p95 on `AgentChatModel` + `SentenceStream` is unchanged (`<= +0.05 s` vs `SUFFICIENCY=False`).
- **H4 (Bounded cost & latency):** Extra verdict calls add `<= 0.35` cheap calls/turn and `<= 0.15` web searches/turn on mixed traffic (`<= $0.25` per 100 turns).

## 2. Dataset & Split Hygiene (`eval/sufficiency/implicit_compound.jsonl`)

Current set (`eval/sufficiency/implicit_compound.jsonl`): 60 rows across 34 files (`needs_web_stale`: 20, `needs_web_missing_half`: 15, `needs_web_conflict`: 10, `docs_only_control`: 15).

- **Additions (+40 rows -> 100 total):** Keep `scripts/check_implicit_set.py` style rules, adding:
  - `+10 wrong_question` (10 off-target/decoy-accepted doc answers across distinct files to test `wrong_question` recall without speaking a web fix);
  - `+10 docs_only_control` with temporal-looking words (`"this year"`, `"current"`, `"FY26"`, `"2025 policy"`) and `file_cards` where the doc *does* cover the requested period (total controls = 25);
  - `+10 needs_web_missing_half` on **undated / current-year** files with `>= 4`-word doc answers (exposing R2-3 gate gaps);
  - `+10 regression_controls` (5 `ask_documents` `status="web"` partial rows to enforce no double-lookup, + 5 ME3 fast/command/pleasantry turns).
- **Independent frozen split (by source document family, never by paraphrased question):**
  - Partition the 34+ corpus files (grouped by directory/agency family, e.g. `irs/*`, `census/*`, `ons/*`, `eia/*` stay together) **60 / 40**:
    - `implicit_dev.jsonl` (~60 rows): used to tune `SUFFICIENCY_SCHEMA`, gate cues, and prompt wording.
    - `implicit_frozen.jsonl` (~40 rows, unseen file families): **sealed until Stage 2 freeze**; evaluated **once** at Stage 2/3 with no further prompt/gate edits afterwards (following `docs/GPT-SOL-CRITIQUE-2026-11-26.md`).

## 3. Metrics & Ship Thresholds

| Metric | Definition | Ship Threshold |
|---|---|---|
| Recall by kind | `stale` / `missing_half` / `conflict` -> non-`sufficient` + usable `web_question`; `wrong_question` -> `wrong_question` | `stale >= 0.85`, `missing_half >= 0.70`, `conflict >= 0.65`, `wrong_question >= 0.70` |
| Control over-trigger | Fraction of `docs_only_control` triggering `web_answer` (`sufficiency_web=True`) | `<= 0.10` (`<= 0.05` spoken extra) |
| First-audio delta | Real `AgentChatModel` + `SentenceStream` first-audio p50 & p95 vs `SUFFICIENCY=False` | `<= +0.05 s` (no delay) |
| Extra-call & cost | Verdict calls/turn, `web_answer` searches/turn, USD per 100 mixed turns | Verdict `<= 0.40/turn`; web `<= 0.20/turn`; `<= $0.25 / 100 turns` |
| Spoken quality (`judge_eval.py`) | Qwen logprob `directness` (`A: answers`), `faithfulness` (`A: supported`), `completeness` (`A: covered`) on final `result["text"]` | Net `completeness` gain `>= +0.20` on `implicit_frozen`; `directness A >= 0.85`; zero retractions / `REPEAT_PREFIX` leaks |

## 4. Experiment Order (Cheapest Signal First)

1. **Stage 0 — Offline unit & contract gates (free):** Fix R1/R2 on `exp/sufficiency-fix`, make every `xfail` in `pipeline/tests/test_sufficiency_review.py` pass (`strict` -> normal pass), verify `pytest`, `ruff`, `mypy`.
2. **Stage 1 — Offline `verdict()` on gold `doc_fact` answers (free, Qwen3.8-27B):** Feed `(question, ideal_doc_answer=doc_fact, source_year, today)` into `should_check` + `sufficiency.verdict` on `implicit_dev` using local Qwen (`http://127.0.0.1:18080/v1`, `json_schema`). Separately report gate recall vs verdict recall. Tune prompt/gate on `implicit_dev` only; validate once on `implicit_frozen` (Qwen). **Do not proceed if Stage 1 fails recall/over-trigger bars.**
3. **Stage 2 — Full end-to-end flow with Qwen answerer + scripted/cached web + Qwen judge (free):** Run full `AgentServer` (`_ask_docs` + `_maybe_supplement` + `SentenceFilter` + `AgentChatModel`/`SentenceStream` loopback) on `implicit_frozen` + `docs/ME4-SPOKEN-QUESTIONS.md` subset using Qwen as both answer/verdict model and `scripts/judge_eval.py` (`directness,faithfulness,completeness`). Verify first-audio latency, zero `REPEAT_PREFIX`, zero `wrong_question` web calls, zero ME3 command regressions.
4. **Stage 3 — One small paid Gemini subset (`<= $0.10` hard cap):** Run ~30 stratified rows from `implicit_frozen` (10 `stale`, 8 `missing_half`, 4 `conflict`, 8 `docs_only_control`) with real `gemini_json_request` (`SUFFICIENCY_SCHEMA`) + real `web_answer` (ledger tracked). Score with `scripts/judge_eval.py` + manual inspection of spoken text and spend ledger.

## 5. Stop Conditions

Abort and leave `SUFFICIENCY=False` if any of the following occur:
- `docs_only_control` false web-supplement rate `> 15%` in Stage 1 or Stage 2;
- `directness A` or `faithfulness A` drops below the `SUFFICIENCY=False` baseline by `> 3` points, or any retraction / `REPEAT_PREFIX` / leaked private ID appears in `result["text"]`;
- `AgentChatModel` first-audio p95 degrades by `> 0.15 s` or any barged-in turn blocks `self.lock` `> 0.5 s`;
- Stage 3 Gemini spend exceeds `$0.10` or projected cost exceeds `$0.30 / 100 turns`;
- Any ME3 `fast` / `routed` / `chain` regression appears in the spoken test bank.

## 6. Rollout & Rollback Checklist

- [ ] **Flag & config:** `sufficiency.SUFFICIENCY = False` default; expose `--sufficiency` / `VOICE_AGENT_SUFFICIENCY=1` in `agent_server.py` (like `--docs-first`) so Pi/laptop configs toggle without code edits.
- [ ] **Schema & budgets:** Dedicated `SUFFICIENCY_SCHEMA` in `gemini_json_request`; `VERDICT_TIMEOUT_S=1.5`, post-speak cap `<= 1.0s` verdict + `<= 2.5s` web, `self.offline()` guard active.
- [ ] **Logging fields (turn JSONL):** `sufficiency` (`sufficient|stale|missing_half|wrong_question|none`), `sufficiency_reason`, `sufficiency_gate` (which leg fired: `time|short|year|none`), `sufficiency_web` (`bool`), `sufficiency_s`, `sufficiency_web_s`, `sufficiency_cancelled` (`bool`).
- [ ] **Canary:** Run the **Core 20** + `g2-x*` + `g2-t*` from [ME4-SPOKEN-QUESTIONS.md](ME4-SPOKEN-QUESTIONS.md) in pure audio on the Pi (`--hedge -1 --docs-first --sufficiency`) and audit `turns.jsonl` + `web_answer_ledger.jsonl` before enabling by default.
- [ ] **Rollback:** Instant config rollback (`VOICE_AGENT_SUFFICIENCY=0` / drop `--sufficiency`, restart `pi-agent`); reverts to `me4-rc-2026-11-27` behaviour byte-for-byte.
