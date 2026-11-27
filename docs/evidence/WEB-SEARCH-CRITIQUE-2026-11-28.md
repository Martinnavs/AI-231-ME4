# Evidence record: grounded web-search critique

Audience document: `WEB-SEARCH-CRITIQUE-2026-11-28.md`. Reader: repository owner deciding how to improve current grounded Gemini. Status: internal technical proposal, 2026-11-28, Asia/Manila. User clarified that grounded Gemini is the current backend; this governs recommendation priority.

## Sources reviewed

Repository: `/mnt/School/PhD/AI231/AI-231-ME4`, branch `qwen/ingest`, pulled fast-forward from `1699c56` to `022b310b42b4c9a24003315b131b63c7fa8f82e5`. Pull introduced documentation only. Existing untracked assessment/configuration files were left alone. No applicable AGENTS.md was found at the repo or its ancestor paths checked.

Implementation references below are relative to this evidence directory at its final repository destination.

- [Web tools](../../src/voice_agent/mcp_servers/tavily_search.py): constants lines 57–117; `research_text` 196–216; `content_words`/`same_question` 243–253; registration/cache 407–443; research retrieval 443–533; directness 558–608; grounded lookup 610–643; alternate grounded research 645–675; provider chain 739–775; backend default 791.
- [Grounding](../../src/voice_agent/mcp_servers/gemini_grounding.py): endpoint 34–36; short-answer prompt 46–58; research prompt 72–78; parsing 109–137; directness prompt 60–70, overlap 345–360, call 363–402; request construction 445–459; ledger 486 onward.
- [Pipeline guards](../../pipeline/src/voice_pipeline/agent_tools.py): `FAILURE` line 258; shared `LOOKUP_TOOLS`/repeat prefix 431–432; `one_fact_question` 1103–1111; redirect 1256–1269; per-turn lookup deduplication and remember callback 1283–1325.
- [Search evaluator](../../scripts/search_eval.py): `passes` checks required substrings; [eight cases](../../config/eval/search_cases.jsonl) lack either example and claim-level temporal/coverage checks. `--dry` always emits all expected keywords, so is transport/scorer plumbing rather than a retrieval benchmark.
- [Miss Universe local transcript](../../out/sessions/stack-1126-1758/turns.jsonl), lines 29 and 31: November 26 at 18:08:06 and 18:08:49. These are ignored local artifacts, not durable tracked evidence. Relevant excerpts are recorded below.
- [KoolPals trace A](../../pipeline/out/agent_server/20261116-104540.jsonl), lines 12–13; [trace B](../../pipeline/out/agent_server/20261116-120321.jsonl), line 7. Also ignored local artifacts.
- [November 26 session findings](../ME4-VOICE-SESSION-FINDINGS-2026-11-26.md): historical Pi deployment and chain-state issues, not assumed current after subsequent fixes.
- [Postdeployment state](../ME4-POSTDEPLOY-STATE-2026-11-27.md): frozen release, experiment branches, owner's order: sufficiency before the benchmark; parked critique fixes. This review does not authorize reordering implementation or modifying frozen code.
- [Existing research handoff](../DEEP-RESEARCH-HANDOFF-2026-11-26.md) and `git show origin/feat/deep-research-skill-harness:src/voice_agent/research/runner.py`: opt-in library on a separate branch; `Source.kind="search_snippet"`, bounds of 30 seconds, four adapter searches, three model calls by default. `src/voice_agent/research` does not exist on this integration checkout. Its own handoff says integration, cost admission, and live quality validation remain.
- [Frozen critique](../FROZEN-ADVERSARIAL-CRITIQUE-2026-11-27.md), web-layer findings: previous directness/negation concerns. Rechecked the relevant implementation and reproduced overlap behavior independently; prior critique labels alone are not fresh verification.

Primary web sources read November 28, 2026:

1. [Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search), redirecting to Google AI Studio documentation. Google says the model decides whether to search and can issue multiple queries. Current documentation describes both newer Interactions output and the legacy `groundingMetadata.webSearchQueries` field; do not copy its Interactions examples into this `generateContent` client. It also says returned search queries are not necessarily the total billed queries; local `Ledger.searches` is therefore an observed-query counter, not a guaranteed exact bill or monetary ceiling.
2. [Google URL Context](https://ai.google.dev/gemini-api/docs/url-context): supports joint use with Search, lists Gemini 3.1 Flash-Lite, provides per-URL retrieval metadata, only retrieves supplied URLs without recursive traversal, and requires public accessible content. Retrieved content adds input tokens; unsupported content includes audio/video and paywalled content. Feature compatibility on the exact configured endpoint has not been exercised here.
3. [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search): `advanced` trades latency for relevance, not guaranteed index coverage; provides domain/date filters and configurable snippets/raw content. No provider configuration changes were made.
4. [Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract): selected-URL raw content and per-URL failures. Proposal only; not currently called by this web server.
5. [Spotify show episodes](https://developer.spotify.com/documentation/web-api/reference/get-a-shows-episodes): episode name/release date/date precision, pagination, market restrictions, and OAuth access. Does not establish this account's endpoint access or that all KoolPals releases are available. Do not assume first response item or largest episode number proves newest eligible release.
6. [KoolPals home](https://thekoolpals.com/home): different Spotify and YouTube release schedules support platform scoping as consequential. No current episode number established. An attempted `/library` open failed; that guessed path is not cited as proof of absence.
7. [Miss Universe homepage](https://www.missuniverse.com/) and [November 25 statement](https://www.missuniverse.com/press-releases/public-statement-7/): the retrieved homepage still labels the titleholder as 2025 while the statement refers to the concluded 2026 edition. This demonstrates why an official homepage alone may be temporally insufficient. It does not verify the 2026 winner and is not a historical capture of what Gemini saw in the incident.

## Claim ledger

| Claim | Support | Status and limits |
|---|---|---|
| Gemini is the user's active backend. | Explicit user clarification; server CLI defaults to Gemini. | User-confirmed. Running service flags and deployed hash were not inspected. |
| Miss Universe got a non-answer and user requested a broader check. | Local row 29's `spoken`: winner reports “currently unavailable”; row 31 asks “could you broadly check the Miss Universe question?” | Observed November 26. Row 29 records barge-in; row 31 lacks a spoken answer field. No claim of no audio whatsoever or a diagnosed November 27 cause. |
| Harness KoolPals answers conflict. | Trace A line 12: 1032 “Stuck Si Roger,” purportedly around Nov 15/16; line 13: Spotify 1053 “Sampal ng Ina,” Nov 5, with no later episode confirmed; trace B line 7: 1056 “Role Sa Teatro,” Nov 12. | Observed generated claims, not verified episode facts. All three call web_research; the first/third also call web_answer. Contradiction is established without choosing a true winner. |
| Grounded non-answer can suppress another backend. | `grounded` returns `(HEDGE, None)`; `search` returns any non-null text. Probe `nonanswer_ends_fallback`. | Reproduced on the real MCP web_answer tool with mocked provider HTTP. Two search-enabled Gemini calls plus two judge calls, zero Tavily calls despite a prepared valid fallback response. Not proved causal for original incident. |
| Pipeline discourages and blocks purposeful research escalation. | Per-turn first-success reuse across LOOKUP_TOOLS; `web_research` redirect on matching one-fact question and inactive chain; tool description says one fact should use web_answer. | Code-verified. “Who won” does not match the specific WHO_QUESTION pattern; “who is” can. No claim that every research query is redirected. |
| Hedge counts as successful lookup for deduplication. | HEDGE does not match `FAILURE`; existing test explicitly enforces that property. | Code-verified; distinction between non-answer and transport failure is absent from the text contract. |
| Source names do not prove the relationship requested. | Parser uses first support index, directness sees generated passage, overlap threshold .6 removes `not`/`no`. Probe `negation_inversion`. | Reproduced helper accepting “Alpha is eligible for the prize” against “Alpha is not eligible for the prize.” This is a guardrail defect, not an observed live false answer rate. |
| Rechecks can be stale or wrongly scoped. | Quick TTL 600 s, research TTL 3600 s, fuzzy Jaccard threshold .75. Probe `platform_cache_collision`. | Reproduced matcher on longer full-length-episode queries differing Spotify/YouTube. Shorter queries in the initial exploratory probe did NOT collide; no universal collision claim. End-to-end wrong-answer replay not performed. |
| Snippet retention can remove decisive evidence. | RESEARCH_SNIPPET=400, RESEARCH_CHARS=2000, include_raw_content=False. Probe `snippet_cutoff`. | Reproduced removal of synthetic winner text present after character 400. Applies to Tavily research path; grounded quick answer is a separate path. |
| Recovery has shallow control and aggressive time bounds. | Three-second Gemini attempt, nominal 4.5-second total; retry appends specificity prompt, retains same question; directness calls have separate 1.2-second budgets. | Code-verified. The nominal budget is not a strict end-to-end bound: directness checks can extend it; multiple Gemini attempts can occur via separate zero-search and directness paths. |
| Gemini-only Search plus URL reading is worth testing first. | User preference; official Google capability docs; absence of URL reading in current request. | Recommendation/inference, no live accuracy or latency benefit claimed. Public URL retrieval may itself use cached/indexed content; it is not guaranteed to bypass every freshness problem. |
| Existing tests cannot establish current-answer quality. | 137 offline tests pass; evaluator is substring-based and eight-case set omits these incidents. | Observed test result and inspected scoring; not a claim that all project evaluations are weak. |

## Contradictions and gaps

- User describes yesterday (November 27); available local Miss Universe voice recording is November 26. The local agent-server directory contains November 16 traces only. Prior session report refers to a November 26 server log on the Pi, not a locally available matching provider trace. No Pi access was attempted for this document task.
- Terra failure has no local transcript, retrieved evidence, query list, budget, or tool configuration. It demonstrates the user's experience, not an isolated indexing cause or a controlled model comparison.
- Current source code and historical deployed code may differ. November 26 report describes an old chain-capture bug; current agent_server excludes ordinary web_answer work when retaining chain state. Do not revive that fixed historical mechanism as a current root cause.
- Fast lookup also falls back to Wikipedia in some failure states. That is useful for settled facts but cannot certify the newest release or event result. A recovery result should distinguish unverified candidate from answered status.
- Live search today cannot reconstruct the index at incident time. No definitive winner/episode gold label is assigned from generated trace text.
- Google documentation now defaults to Interactions examples, while the repo uses REST generateContent and legacy grounding metadata. Exact supported combination and response parsing need an integration test. URL Context alone does not guarantee access to Spotify's rendered/authenticated listing.
- No paid model calls, API-key reads, service restarts, implementation changes, or deployments were performed. Built-in web browsing checked primary public docs/pages only.
- Proposed recovery budgets are trial values, not approved cost/latency requirements. The owner's documented priority order and release freeze remain applicable to any future implementation.

## Verification

Commands run from the repository unless stated otherwise:

```sh
git -c safe.directory=/mnt/School/PhD/AI231/AI-231-ME4 pull --ff-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/mcp_servers/test_tavily_search.py tests/mcp_servers/test_gemini_grounding.py tests/test_search_eval.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python docs/evidence/web_search_audit_probes.py
```

The probes were authored and run at `/tmp/web_search_audit_probes.py`; the last command uses their final preserved path. They use fake keys, httpx.MockTransport, and Ledger(None), with no real provider requests. Sandboxed MCP/test runs stalled; escalated offline reruns completed. The test command was wrapped in `timeout 90`, probes in `timeout 30`.

Result: **137 passed**, one dependency Authlib deprecation warning, **6.80 s**. Four final probe assertions pass:

```json
{"probe":"nonanswer_ends_fallback","reply":"I couldn't confirm that with a search, so I won't guess.","google_requests":4,"tavily_requests":0}
{"probe":"platform_cache_collision","same_question":true}
{"probe":"negation_inversion","supported":true}
{"probe":"snippet_cutoff","winner_in_input":true,"winner_in_output":false}
```

The probe prints Google HTTP requests, not four searches: two are judge calls. Its assertions describe current undesirable behavior so the audit is reproducible; they are not future regression tests that should remain green after a fix.

Document validation: bundled `validate_doc.py` with `--technical` against the document/evidence pair: **PASS**, 1,141 visible words, six second-level sections. Relative links checked using staging links to the inspected repository. Reviewed warnings deliberately: length stays below the 1,200-word limit; status-language warning refers to a proposed structured status and an explicitly owner-approved future run, not a claim that implementation or paid testing is approved. No length or section exception is required.

## Review checklist

- [x] The opening gives the audience enough context without a generic purpose section.
- [x] Every second-level section answers a question specific to this document.
- [x] Every retained detail changes interpretation, a decision, a responsibility, or a next action.
- [x] Prose carries the explanation; every list, table, or diagram has a clear advantage over prose.
- [x] Implementation references are confined to this evidence record except intentional tool names in the technical audience document.
- [x] Terminology is consistent and necessary technical identifiers are explained.
- [x] Claims using exact status words have evidence for that exact status.
- [x] Uncertainty is sparse, consequential, and handed off through the controlled final section only when needed.
- [x] Repetition and generic template sections have been removed.
