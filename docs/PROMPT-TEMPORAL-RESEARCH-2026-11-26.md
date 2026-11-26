# Bounded evidence repair and temporal prompt contract

Research note for the current `ask_documents` / Gemini tool stack. Recommendation only; no implementation. The code observations below are from the current checkout (HEAD `8dbc4e3`).

## Current prompt and data contract

- `ask_documents.SYSTEM` (`pipeline/src/voice_pipeline/ask_documents.py:114-129`) already says documents are untrusted data, the answer must come only from passages, `[Now: ...]` can compute relative dates, and model output must include a short quote. It does not define a bounded verify/repair step or typed time fields.
- `REPLY_SCHEMA` (`:131-142`) only requires `status`; the remaining evidence fields are optional at schema level. The actual deterministic checker does the enforcement afterward.
- `FALLBACK_SYSTEM` (`:144-148`) chooses one query or file from a filename list. This is a retrieval retry, not an answer verification loop.
- `GENERAL_SYSTEM` (`pipeline/src/voice_pipeline/general.py:14-20`) handles folder/catalogue questions only and has no temporal semantics.
- `system_agent.txt` says “one question needs one lookup” and “No, I mean ... redo that action with the correction, make no new lookup.” This is intended to prevent duplicate external actions, but conflicts with any general instruction to retry after a challenged answer. Repair must be scoped to read-only evidence checks; irreversible home actions must never be replayed.
- `AgentServer._ask_docs` (`pipeline/src/voice_pipeline/agent_server.py:703-740`) calls `ask_documents` without `now_stamp`, although `ask_documents` accepts it and `SYSTEM` references it. Meanwhile, the top-level agent has its `[Now: ...]` stamp. This creates prompt-path inconsistency. Current docs temporal handling uses a source-year heuristic to annotate a citation (`ask_documents.py:978-985`), not to establish when the fact was true.
- `_verdict` sets `remaining = budget_s - offset` (`ask_documents.py:855-870`), where `offset` counts fallback calls rather than elapsed seconds. A model request and web request can each consume that same nominal `remaining` (`:988-992`), so it is not a strict wall-clock budget. Any bounded repair proposal needs one monotonic deadline shared by retrieval, generation, verification, repair, and web.

## Recommendation: mini ReAct as evidence repair, not self-critique

Use a small “draft → external check → one repair or abstain” protocol. It borrows ReAct's action/observation pattern and CRITIC's tool-interactive feedback premise, but does not ask the model to privately introspect until it feels certain. The verifier's signal must come from deterministic checks or an external source, not another unsupported opinion from the same model. Huang et al. found intrinsic self-correction without external feedback can fail or degrade reasoning, while CRITIC explicitly grounds correction in tool interaction. ReAct demonstrates alternating reasoning with environment actions; it does not establish that free-form chain-of-thought is a reliable verifier.

Suggested static prompt module for document answers:

1. **Draft contract:** produce a short claim, source file, exact evidence spans, and typed temporal relation if time is involved. Keep user-facing prose separate from evidence fields.
2. **Check contract:** the host checks quote spans, cited file, required answer entities/numbers, row identity where available, arithmetic expression and result, date relation, and whether all requested parts have evidence. A checker failure is a specific signal (`wrong_row`, `missing_claim_evidence`, `calc_mismatch`, `time_mismatch`, `quote_mismatch`, `partial`).
3. **Repair contract:** only on a bounded, recoverable failure and while deadline remains, send the original question, candidate, exact checker error, and relevant retrieved passages back once. Ask for corrected structured output or `abstain`; do not let the model revise the check result or invent new evidence. Re-run the same deterministic checks; on second failure return a clear miss/partial answer.
4. **Retrieval action:** permit one targeted retrieval expansion only for `no_relevant_passage` or a specific missing claim. Do not re-search on purely semantic checker disagreements unless a distinct source or calculation is needed.
5. **Safety boundary:** the loop is read-only. Never repeat a home command, publish, delegate, send a message, or other side effect as part of answer repair. Keep the system's existing “one question, one lookup” rule for ordinary lookup behavior; explicitly define the repair as a single correction to that lookup, not a new unconstrained search.

Initial budget to evaluate (not a performance promise): at most one repair generation, one targeted retrieval expansion, and one total monotonic deadline (e.g. the existing document-turn budget, not a fresh full budget per stage). Trigger only for measurable checks and only when the likely fix can change the result. Track calls and elapsed time per stage. Compare (a) no repair, (b) one tool-grounded repair, and (c) a second-model critique only if it reads the independent evidence and returns a constrained issue code. Do not add an unconstrained “check your work” prompt by itself.

## Temporal contract: distinguish event time from source time

Relative expressions should be resolved against an explicit per-turn `as_of` timestamp and timezone, but the answer must distinguish at least these clocks:

| Field | Meaning | Example |
|---|---|---|
| `event_time` | When the event happened or is scheduled to happen | Meeting occurred 2025-11-02; event planned for 2027-04 |
| `valid_time` (`valid_from`, `valid_to`) | Interval in which a proposition is true in the modeled world | Rate applies from 2026-01-01 through 2026-12-31 |
| `published_at` | When the source published the record | Agency announcement published 2025-12-15 |
| `retrieved_at` | When this system fetched/ingested the source | Retrieved 2026-11-26T12:00+08:00 |
| `recorded_at` / `ingested_at` | When this corpus first stored the record | Indexed at ingest run timestamp |

Publication/retrieval date is not the fact's validity date. A report published in 2026 can discuss a 2024 population estimate; a file modified yesterday can contain a 2022 policy. If validity is unknown, represent it as unknown and say what the source actually dates. Do not infer “current” from file mtime or retrieval time.

Normalize question intent into a small relation before generation:

- **Past:** “What was X in 2024?” requires evidence whose event/valid time covers 2024; a later retrospective source can be valid evidence if it explicitly reports that period.
- **Current:** “What is X now?” requires a source valid at `as_of`, preferably an authoritative, recently updated source. If evidence is only historical, label its period and do not promote it to current.
- **Future:** “What will X be next year?” is a forecast or scheduled future fact, not an extrapolation from the current document. Use an official announced future value when present; otherwise say it is not yet established. “Next year” is computed from `as_of` in the user's timezone.
- **Relative event date:** “last Tuesday” and “in two weeks” resolve from `as_of` and timezone; preserve the resolved ISO date in the check record so DST/calendar interpretation is reviewable.
- **Change over time:** require both endpoints and units, and compute only after verifying each endpoint refers to the requested entity, measure, and period.

Dynamic state should be per-turn and ephemeral: `as_of`, timezone, requested temporal relation/interval, retrieved evidence metadata, and source-selection rationale travel with the current request/check trace. Do not persist a model-inferred “current truth” in conversation memory or the document index. Persist only explicit source metadata and ingestion timestamps, with provenance. A cache key for a time-sensitive answer must include a suitable freshness boundary or be bypassed; the current normalized-question doc cache is per-turn on the server, which is safer than cross-turn storage.

## Source selection and conflict resolution

Use a source priority appropriate to the claim, not a universal web ranking:

1. For **what a user's document says**, cite that document and its reported period. Do not replace its historical contents with web data.
2. For **current public facts**, prefer the responsible primary source (official agency, organization, legislation, dataset, release notes), with publication/update date and validity period. Use reputable independent reporting if primary evidence is absent, inaccessible, or insufficient.
3. For **historical event facts**, prefer contemporaneous primary records where available; later official summaries are acceptable when they clearly identify event date and scope.
4. For **forecasts/plans**, distinguish announced/scheduled from realized. Prefer the issuing body and retain modality (“planned,” “projected,” “proposed”).
5. On material disagreement, do not silently average or select by recency alone. Return a short conflict/uncertainty statement naming the different periods or sources and ask a clarifying question if the entity/measure is ambiguous.

## Minimal experiment set

Use a held-out, human-labeled set with matched past/current/future questions over the same facts; include source publication date ≠ validity date, stale corpus facts, future plans later changed, ambiguous timezone, false premise, mixed document+web answer, and conflicting official sources. Measure: factual correctness; date/interval correctness; source attribution; unsupported freshness claims; correct abstention; repair success and regressions; total calls; P50/P95 elapsed time. Compare the bounded loop against no repair. Count gains only if the independent verifier catches a specific error and the repaired final output passes the same checks.

## Primary sources

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): reasoning/action interleaving with external environment observations; benchmark evidence, not proof that unconstrained self-critique is reliable.
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing (Gou et al., 2023)](https://arxiv.org/abs/2305.11738): correction follows feedback from external tools; supports evidence-conditioned checking rather than blind reflection.
- [Large Language Models Cannot Self-Correct Reasoning Yet (Huang et al., ICLR 2024)](https://openreview.net/forum?id=IkmD3fKBPQ): reports that self-correction without external feedback can fail or degrade performance across tested tasks/models; motivates bounded external verification.
- [FreshLLMs / FreshQA (Vu et al., Findings of ACL 2024)](https://aclanthology.org/2024.findings-acl.813/): dynamic benchmark separating never-changing, slow-changing, fast-changing, and false-premise questions; supports testing temporal categories separately and refreshing labels over time.
- [W3C OWL-Time](https://www.w3.org/TR/owl-time/): standard vocabulary for instants, intervals, and temporal relations; useful for representing event/valid intervals, not itself a source trust policy.
- [W3C PROV-O](https://www.w3.org/TR/prov-o/): provenance vocabulary for entities, activities, agents, and generation/use relations; can distinguish source artifacts and retrieval/ingestion activities from claims about world-valid time.

The research supports the design pattern, not a guaranteed accuracy or latency gain. The proposed one-repair cap, triggers, source ranking, and temporal fields are engineering hypotheses to evaluate against this stack.
