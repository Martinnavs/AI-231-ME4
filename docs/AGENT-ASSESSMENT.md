# Agent assessment: how it works, every check it makes, what is missing, and how to improve it for RAG

Written 2026-11-18 on branch `text-only`. This is the master document for improving the front agent and preparing it for RAG over files the user ingests. It describes the code as it stands; no code was changed to write it. Two independent assessments by other models (agy / Gemini 4 Argon High, codex / gpt-6.1-terra xhigh) were run from the same brief; their full reports and a claim-by-claim validation are in `docs/agent-assessment/` (start with its `README.md`); section 13 compares them with this one.

**How to read the citations.** Code is `path:line`, paths from the repository root. Readings use the short keys in section 14 (for example **[CE-M1 §6]** is the Context Engineering course, module 1, concept 6; **[BOOK p.92]** is *Hands-On RAG for Production*, page 92). Numbers from logs come from `pipeline/out/agent_server/*.jsonl` (turn rows), `~/.local/state/voice_agent/web_answer_ledger.jsonl` (grounded calls) and `~/.local/state/voice_agent/gate_outcomes.jsonl` (second looks). Anything not verified is marked **unverified**.

## 1. Bottom line

1. **The web-fact path is strong at the lookup and weak after it.** A changeable question is classified (EverGreen, FreshQA recall 0.80, `src/voice_agent/gate.py:1-10`), forced to search, falls back to Tavily, and abstains instead of guessing (`src/voice_agent/mcp_servers/tavily_search.py:558-590, 686-721`). But a grounded answer that ran at least one search is accepted with no check that the answer is supported by what was found, or that it answers the question (`tavily_search.py:581-583`). In the readings' terms the agent checks **process** (a search ran) and not **outcome** (the answer is grounded and relevant) **[EV-M4 §4]**, **[BOOK p.160-161]**.
2. **Several checks do not reach the turns they were written for.** The "this can have changed" note (`pipeline/src/voice_pipeline/call_policy.py:668`) is only appended by `steer()`, and the deployed configuration (router + `--hedge 0`) sends ordinary turns through `stamped()`, which adds only the date line (`pipeline/src/voice_pipeline/agent_server.py:587-598`). The pipeline's own changeable test is the regex (recall 0.55), not the EverGreen gate the lookup uses (recall 0.80) (`call_policy.py:643-661`, `docs/TEXT-MODE.md` "The changeable-question gate").
3. **The hedge race duplicates work and the logs cannot say who did what.** In 40 of the 54 logged turns of 2026-11-16, one turn made two or more lookup calls: the router's document race (`search_docs`), the router's `web_answer`, then Gemini's own lookup after it won the race. Rows record calls with no attribution to the racer.
4. **The document path is not ready for RAG.** BM25, k=3, 700-character chunks over a fixed corpus (`src/voice_agent/mcp_servers/docs.py:24-46`); no file dates, hashes, versions or ingestion; the verbatim-quote check exists only on the router's race path (`agent_server.py:336-362`), not when Gemini calls `search_docs` itself (Case C, `pipeline/config/eval/system_agent.txt:20`). Of the spec's D1-D8, none is done.
5. **Evaluation exists but is fragmented and small**: 8 search cases, 23 chain scenarios, 105 dry-run prompts, unit tests on hand-built fixtures (CR-1 open). There is no stratified benchmark, no groundedness or answer-relevance evaluator, no retrieval metric, no trace. The readings are unanimous that this comes first **[EV-M1 §3]**, **[AAI-M4 §1, §8]**, **[BOOK p.157]**.
6. **The independent reviews added nine verified problems** (F13-F21, section 7). The most serious: a research chain survives a server restart and steers the next session's unrelated questions (F13), and the document checker accepts an answer that contradicts its own quote (F16).
7. **Order of work** (section 12): traces → benchmark with single-purpose evaluators → close the post-lookup gaps (support, relevance, the note and gate reaching every turn, the race) → ingestion with dates and versions → retrieval measured, then improved → memory.

## 2. One turn, end to end

```
user text (text_chat: last 4 exchanges, text only; src/text_chat/cli.py:34-37, 80-82)
  -> AgentServer.handle (agent_server.py:723)
       backchannel / apology ............ empty reply              (:798-809)
       fast path (lights, clock, ...) ... fixed confirmation       (:810-811, fast_path.py)
       chain active ..................... Gemini + steer(case_f)   (:812-819)
       router configured ................ routed_turn              (:820-826)
            hedged (production): router and 2 Gemini requests race; Gemini gets stamped() only (:587-598, hedge.py)
            not hedged: router first; Gemini only on escalation, with steer(case_f) (:599-610)
       otherwise ........................ Gemini + steer           (:827-831)
  -> output filters: PayloadFilter, SourceFilter, SentenceFilter (:732-751; payload_guard.py, spoken_filter.py)
  -> server source sentence if the reply dropped the lookup's source (:851-861; agent_tools.py:382-403)
  -> background second look (web_verify) on the lookup's claim (:868-869, :612-640)
  -> turn row to pipeline/out/agent_server/<start>.jsonl (:870-890); chain bookkeeping (:891-897)
```

**Production configuration in text mode.** `scripts/stack/up.sh:152-155` starts the agent with `--hedge 0 --router-url ...` (or `--hedge -1` without the router). With `--hedge 0`, `run()` builds a `HedgedAgent` of two `Agent`s (`agent_server.py:983-997`) and the router as a third competitor (`hedge.py:94`). Model: `gemini-3.1-flash-lite` (`agent.py:53`), turn timeout 20 s (`agent.py:61`).

### 2.1 What the model sees on a turn

| Part | Content | Where |
|---|---|---|
| System prompt | `system_agent.txt` (7,934 bytes: Case A-G, `say` rule, attribution rule, "You cannot" block, the Now-line rule) + profile rule, owner name, venue line, publish rule when configured | `system_agent.txt:1-38`, `agent_server.py:943-959` |
| Tools | about 10 schemas, each with an added `say` argument; music hidden; delegate status/cancel hidden until a job exists | `agent_tools.py:1266-1272`, `agent_server.py:965-966, 759-762` |
| History | the client's last 4 exchanges, spoken text only; the server resends them each turn (`reset=True`) | `cli.py:34-37`, `agent_server.py:544-546`, `agent.py:418` |
| Tail of the last user message | hedged turns: `[Now: <date, time>, Asia/Manila. Place: <venue>.]` only. `steer` turns: chain block, Case F protocol (2,035 bytes), the changeable note, then the Now line | `agent_server.py:241-272, 900-904` |
| Router | 1 earlier pair, Gemini replies cut to 100 characters | `router.py:29`, `agent_server.py:225-239` |

The ordering is cache-friendly (static system, volatile tail last) **[CE-M4 §4]** (context window layout), but cached tokens are never measured: `Agent(...)` is built without `usage=True` (`agent_server.py:980, 987`; default `agent.py:373`).

## 3. Every check on the way to an answer

### 3.1 Before the model

| Check | What it guarantees | What it does not | Code |
|---|---|---|---|
| Backchannel / apology filter | "mm-hmm", "sorry" start nothing | only exact short forms | `agent_server.py:798-809`, `spoken_filter.py:44-47, 71-78` |
| Fast path | common home commands and the clock skip the LLM | context-free except "turn it up" after a temperature/brightness turn | `fast_path.py`, `agent_server.py:274-282, 773-777` |
| Router validity (`call_policy.decide/plan`) | a routed call has the slots the user said; lookups whose question needs the conversation go to Gemini | the router sees one earlier pair | `call_policy.py:534-761, 687-713` |
| Offer answer | "yes" to the agent's offer goes to Gemini, not the router | | `call_policy.py:52-69`, `agent_server.py:824` |

### 3.2 Lookups (the `web_answer` MCP tool and its wrapper)

| Check | What it guarantees | What it does not | Code |
|---|---|---|---|
| Changeable gate (EverGreen int8 ONNX, 4 ms, threshold 0.5; regex fallback) | a changeable question is recognised (FreshQA recall 0.80, false alarms 0.18) | 20 % of changeable questions are missed; only the MCP child has it | `gate.py:1-10, 24-26, 82-90`, `gemini_grounding.py:139-144` |
| Grounder prompt | the date and place; "anything that can change needs a search result" | it is an instruction, not a check | `gemini_grounding.py:47-55` |
| Forced retry | a changeable question answered with zero searches is asked again with "Search the web before you answer" within a 4.5 s budget, skipped under 1 s left | | `tavily_search.py:569-580, 70-71`, `gemini_grounding.py:61` |
| Tavily fallback | when grounding fails or never searched | Tavily's own generated `answer` is spoken; see `unconfirmed` | `tavily_search.py:686-716, 230-240` |
| `unconfirmed()` | every number and capitalised name **of the question** appears in the answer or the results, else a hedge | it checks question → evidence, not answer → evidence: an answer's own invented name passes | `tavily_search.py:256-286` |
| Abstaining hedge | a changeable question nothing could check gets "I couldn't confirm that with a search, so I won't guess.", not cached | | `gemini_grounding.py:321-326`, `tavily_search.py:719-720` |
| No guess cached | the unsearched first text is never cached | a searched answer is cached 600 s by content words (Jaccard ≥ 0.75) | `tavily_search.py:74, 86, 421-424, 535-538` |
| In-flight join | the same question asked twice shares one request | | `tavily_search.py:425-431` |
| Per-turn lookup dedupe | a second lookup in one turn gets the first result with "(already looked up this turn)"; a cancelled first lookup no longer yields an empty "already looked up" | | `agent_tools.py:1054-1099` |
| `web_research` → `web_answer` redirect | "who is ..." single facts use the grounded answer, keeping the model's query when the user's words refer back | | `agent_tools.py:1001-1040` |
| Failure detection | a reply matching `couldn't find|did not answer|not connected|can't say|error` is a failed lookup: retried once if transient, handed back to the model | **any** answer containing "error" or "couldn't find" (news about an error) is treated as a failure (CR-4) | `agent_tools.py:185-187, 1242-1249`, `agent_server.py:329-334, 479-489` |
| Spoken-as-is | `web_answer` results are spoken without a second model pass unless the question asks for a decision | | `agent_tools.py:172-182, 1239-1251`, `question.py:27-29` |
| Untrusted research text | `web_research` results are wrapped in `<untrusted_search_data>`, HTML stripped, lines with "ignore previous" etc. removed | a four-string blocklist | `tavily_search.py:180-216` |
| Budgets | grounded searches capped at 4,500 a month; Tavily modes normal/conserve/reserve | | `gemini_grounding.py:45, 370-389`, `tavily_budget.py:19-35` |

### 3.3 Attribution

| Check | Guarantees | Does not | Code |
|---|---|---|---|
| `attribute()` lead-in | "As per <publisher>, ..." from the grounding chunk behind the first supported claim; "(the only source I found)" for a changeable fact with one site | the publisher is derived from a title or domain by rule (`AbsCbn`, `Gmanetwork` were seen, R4) | `gemini_grounding.py:111-122, 182-250` |
| Prompt rule | keep the "As per" opening; "reportedly" and the source for what a named person did | instruction only; unmeasured (S8) | `system_agent.txt:34` |
| Server backstop | if the spoken reply names none of the lookup's sources, the server sends "Source(s): ..." | recognises only "according to / per / from X" or "X says" | `agent_tools.py:382-403`, `agent_server.py:851-861` |
| `SourceFilter` / `strip_sources` | "(Source: ...)" asides are not spoken | | `payload_guard.py:58-110` |

### 3.4 After the turn

| Check | Guarantees | Does not | Code |
|---|---|---|---|
| Second look (`web_verify`) | for a changeable question, a forced-search re-ask compared by a SAME/DIFFERENT call; DIFFERS becomes a notice; capped (100 in text mode) and counted in a file | only after lookups with a source mark; not after research answers (S12); the comparison is one flash-lite word | `tavily_search.py:623-676`, `gemini_grounding.py:56-60, 266-309`, `agent_server.py:612-640` |
| `unchecked` audit flag | a turn on a changeable question with no lookup is flagged in the log | uses the regex gate | `agent_server.py:875-884` |
| `output_audit.py` | offline checks: repeated sentences, premature "I have updated", duplicate jobs, fragment-triggered tools | no groundedness or relevance check | `scripts/output_audit.py` |

### 3.5 Documents

| Check | Guarantees | Does not | Code |
|---|---|---|---|
| BM25 retrieval | top 3 of 700-character chunks (10 % overlap), with the file name | no metadata (date, hash, version), no hybrid, no rerank, corpus fixed at start | `docs.py:24-46` |
| Doc race (router path only) | when the router picks one `web_answer`/`wiki_summary`, the documents are searched and a 2.5 s Gemini call must answer with a verbatim 3-12-word quote whose names and numbers are in the passages; else the web answer | it is a race: the first acceptable wins, documents and web are never compared | `agent_server.py:336-405, 456-471`, `doc_answer.py:19-27, 53-72, 109-123` |
| Case C (Gemini calls `search_docs`) | the prompt says answer from the passages or say you don't know | **no quote check, no names/numbers check**, passages not marked untrusted | `system_agent.txt:20`, `agent_tools.py:1041-1043` |
| Own-work guard | questions about the agent's own work are answered "not in documents" | | `agent_tools.py:1041-1043`, `call_policy.py:434-441` |

### 3.6 Delegate (background builds)

Code-enforced: refuses publish/push tasks, ignores acknowledgements, asks before destructive tasks, injects chain facts marked "unchecked", primary-source URLs and the NOTES.md sentence, update-in-place sentence, manifest requirement, a 1.5 s grace, replacement of a job restated within 30 s (`agent_tools.py:1116-1211, 191-292`). `NOTES.md` is headed as untrusted data and lives 600 s (`scratchpad.py:18-24`).

## 4. Context and memory

| Layer | Now | Code |
|---|---|---|
| Within a turn | tool results enter the model's context; `web_answer` is usually spoken as is | `agent_tools.py:1239-1251` |
| Within a session | 4 text exchanges; no tool calls, sources or job ids; notices take a slot as a fake user turn (doc_chat) | `cli.py:34-37`, `docs/CONTEXT-CRITIQUE.md` D6, D11 |
| Task state | `ChainState`: phases, up to 8 research facts, 12 items, 6 gaps; ends after 600 s idle or 3 unrelated turns (D2 double count fixed: one site, `agent_server.py:897`) | `chain_state.py:18-37, 233-243` |
| Scratch | NOTES.md for the delegate only | `scratchpad.py` |
| Across sessions | the owner profile (`config/profile.md`); nothing learned is read back | `agent_server.py:944-952` |

By **[CE-M3 §3, §6]** the session window is plain truncation with no compaction and no scratchpad for the front agent; by **[CE-M4 §1, §3]** and **[LETTA Lab 3]** the profile is the only "core memory" block and there is no "archival" store.

## 5. Evaluation and observability today

| Asset | Size / state | What it measures |
|---|---|---|
| Unit tests | root ~1,389 passed, pipeline 771 passed at the last review (`.scratch/context-p1-p3/REVIEW.md`, not re-run for this document) | code paths; grounding gate tests use hand-built responses (`tests/mcp_servers/test_gemini_grounding.py`, CR-1) |
| `config/eval/search_cases.jsonl` + `scripts/search_eval.py` | 8 cases | keywords in a lookup result (live or `--dry`) |
| `config/eval/chain_scenarios.jsonl` + `scripts/chain_eval.py` | 23 scenarios | chain mechanisms with randomised parameters, cost cap |
| `pipeline/config/eval/agent_prompts*.jsonl` + `dryrun.py` | 90 + 15 prompts | tool selection and slots against expectations |
| `scripts/freshqa_gate_check.py` + `tests/fixtures/freshqa_sample.csv` | FreshQA | gate recall and false alarms |
| `scripts/capture_grounding.py` | written, never run (needs approval) | real grounding responses as fixtures |
| Turn log | one JSON row per turn: question, calls, timings, hedge fields, router lines | no session id, no racer attribution, no tokens, no result or sources per call; second-look rows share the file |
| Ledger | 418 rows; query logged since 2026-11-16 09:54 (56 rows) | searches, tokens, seconds, sources; no turn id to join with the turn log; `grounded_research` rows have no query |
| Gate outcomes | 11 second looks: 8 SUPPORTED, 3 UNVERIFIED, 0 DIFFERS. Rows 1-5 are SUPPORTED with **0 searches** (before the rule that a check must search, `tavily_search.py:647-652`); only rows 9-11 are searched SUPPORTED | historical rows mix code versions |

Against **[EV-M1 §6]** (spans, traces, sessions), **[EV-M2 §4]** (stratified benchmark, ≥ 50 % hard cases), **[EV-M3 §1-§4]** (code evaluators first, single-responsibility judges validated by kappa ≥ 0.7), **[EV-M4 §1-§4]** (tool selection, invocation, required presence, forbidden absence; process vs outcome) and **[BOOK p.168-177]** (retrieval and generation metrics), the gaps are: traces, a benchmark that mixes regression, happy-path and hard cases, a groundedness evaluator, an answer-relevance evaluator, retrieval metrics on golden passages, multi-run pass rates, and token and cost accounting.

## 6. Measured evidence

| Observation | Source | Reading |
|---|---|---|
| 40 of 54 turns (2026-11-16 sessions) made ≥ 2 lookup calls, typically `search_docs, web_answer, web_answer/web_research`; the rows carry `hedge_winner` and `router_lines` `web_answer|...`, so the router's race ran and Gemini then made its own call. News turns took 4.4-6.4 s | `pipeline/out/agent_server/20261116-*.jsonl` | these were probe sessions, so the rate is biased; the pattern is not |
| "who is the ceo of nvidia" → "Jensen Huang is no longer the CEO ... reports regarding his potential return"; the user had to restate | `20261116-104540.jsonl` | answer relevance failure **[BOOK p.161]** |
| 9 of 56 queried grounded rows ran zero searches: 5 settled ("capital of France" ×4, Galileo), 4 changeable. For 2 of the 4 a forced row with 1 search follows within 3 s; one (Microsoft CEO) has no forced row; one ("nvidia turmoil" news) is followed by a research row | ledger | the retry fires but not always; cause of the Microsoft miss **unverified** (no turn id) |
| All 5 forced rows searched (≥ 1) | ledger | partial live evidence for CR-1 assumption (a) |
| Median 1.73 s with a search, 1.49 s without | ledger, 56 rows | the search costs about 0.25 s here |
| The second look returned SUPPORTED 8, UNVERIFIED 3; 5 of the 8 SUPPORTED ran no search (older code) | `gate_outcomes.jsonl` | small sample; only 3 searched checks |
| A research chain from the 12:03 session (`chain_update` KoolPals, 12:03:43) was loaded by the server started at 12:07:40; the first turn ("who has arrived for the ASEAN summit") went to Gemini with `[ACTIVE TASK: phase=researching ...]` and no router; the next turn (Nvidia news) carried the ASEAN chain | `out/stack/logs/agent.log:240-282`, `20261116-120321.jsonl`, `20261116-120744.jsonl` (no `router_lines`) | stale task state crosses a restart (found by agy, verified here) |
| In the 10:45 session all 4 second looks returned `SKIP: checks are off` | `20261116-104540.jsonl` | the check is configuration-dependent; off unless `VOICE_AGENT_VERIFY_MAX` is set |

## 7. Problems, ranked

Severity × frequency per **[EV-M2 §3]** (fix rare-harmful before frequent-cosmetic). Each problem has the taxonomies' labels: agent failure mode **[EV-M2 §3]**, context failure mode **[CE-M1 §4]**, RAG failure stage **[BOOK p.157-163]**.

| ID | Problem | Evidence | Labels | Sev. | Freq. |
|---|---|---|---|---|---|
| F1 | A searched grounded answer is spoken without a support or relevance check | `tavily_search.py:581-590` (the comment records that a names check hedged a correct answer once) | Groundedness; generation faithfulness / answer relevance | High | every web answer |
| F2 | The changeable note never reaches hedged routed turns; the pipeline judges "changeable" by the regex (0.55 recall) while the lookup uses EverGreen (0.80) | `agent_server.py:256-257, 587-598`; `call_policy.py:643-661` | Execution (instruction absent); context confusion between two definitions | High | most turns in production config |
| F3 | Hedge race duplicates lookups; the document race result is discarded when Gemini wins; no attribution in logs | section 6 row 1; `agent_server.py:562-598`, `hedge.py:28-30` | Execution (redundant steps); cost/latency **[AAI-M4 §14-15]** | Medium | 40/54 probe turns |
| F4 | Case C document answers have no quote or names/numbers check and no untrusted marker | `system_agent.txt:20`; checks only in `doc_race` | Groundedness; E6 | High for RAG | every Gemini `search_docs` |
| F5 | No document metadata, versions or freshness; documents and web race instead of being compared | `docs.py:24-46`; `agent_server.py:364-405` | Context poisoning / clash **[CE-M1 §4]**; ingestion staleness **[BOOK p.162]** | High for RAG | n/a yet |
| F6 | `FAILURE` regex treats legitimate answers containing "error" / "couldn't find" as failed lookups | `agent_tools.py:185` (CR-4) | Execution | Medium | rare |
| F7 | `unconfirmed()` checks the question against the evidence, not the answer's claims | `tavily_search.py:256-286` | Groundedness | Medium | Tavily fallback only |
| F8 | No traces: no session id, no racer attribution, no tokens, no per-call results; ledger cannot be joined to turns | section 5 | observability **[EV-M1 §6]** | High (blocks everything) | always |
| F9 | Grounding tests rest on hand-built responses (CR-1) | `.scratch/context-p1-p3/REVIEW.md` | test validity **[EV-M3 §1]** | Medium | n/a |
| F10 | History carries what was said, not what was done; nothing across sessions | section 4 | Completion / propagation across turns **[EV-M4 §3]** | Medium | long sessions |
| F11 | Prompt grows by one rule per incident (Cases A-G, many "never"s; CR-2 shows a moved rule blocking a corrected lookup) | `system_agent.txt` | Context confusion / distraction **[CE-M1 §4]**; over-correction trap **[EV-M5 §4]** | Medium | ongoing |
| F12 | Injection defence for web/doc text is a four-string blocklist plus a wrapper (research only) | `tavily_search.py:183-194` | E6 **[BOOK p.86-88]** | Medium for RAG | rare |

Added after the independent reviews (section 13); each was checked against the code or logs before it was listed:

| ID | Problem | Evidence | Labels | Sev. | Found by |
|---|---|---|---|---|---|
| F13 | `chain.json` survives a server restart within 600 s and steers the next session's unrelated turns, bypassing the router | section 6; `chain_state.py:156-175` (load on start), `agent_server.py:812-819` | Context poisoning / clash; propagation | High | agy |
| F14 | The per-turn lookup reuse ignores the query and the tool: any second `web_answer`/`web_research`/`wiki_summary` in a turn gets the first result; a failed first lookup resolves the shared future with its failure text, and a waiter returns "(already looked up this turn) <failure>" | `agent_tools.py:1054-1070, 1094-1096` | Execution; E7 | Medium-high | codex, agy |
| F15 | (reproduced, VALIDATION §4) The cross-turn cache key drops tense and time words (`who, is, was, current, now, today` are stop words), so "who is" and "who was the CEO of Nvidia" share a cached answer for 10 min | `tavily_search.py:87-95, 243-254, 421-424` | Propagation; retrieval | Medium | codex |
| F16 | (reproduced by running `judge()`, `docs/agent-assessment/VALIDATION.md` §4) The document checker accepts a contradicting answer ("Alice is not the CEO" passes against the quote "Alice is the CEO"); digits are matched as substrings ("5" is found in "450"); "158,000" does not match "158000"; `passages_of` drops the file name, so a document answer cannot name its file | `doc_answer.py:39-41, 60-72, 75-93` | Groundedness; S9, S11 | High for RAG | codex (negation, file name), agy (digits, commas) |
| F17 | Tavily `web_research` results carry no "As per"/"Sources:" marks, so the server's source backstop never fires on them; the backstop is also skipped whenever the reply contains the substring "source" ("sources including ...") | `tavily_search.py:196-216`, `agent_tools.py:366-403` | Groundedness (attribution); S7 | Medium | agy |
| F18 | A correction notice is shown in the chat but never added to the history sent next turn, and notices are drained only before the input prompt, so a correction appears after the user's next message | `src/text_chat/cli.py:63-82` | Propagation; E5 | Medium | codex, agy |
| F19 | The pipeline's changeable test requires a question word at the very start: "hello, can you tell me who ..." is not changeable to it | `call_policy.py:650-661` | Execution | Medium | agy |
| F20 | The Tavily hedge "I couldn't confirm X in what I found. The closest result says: ..." still speaks the unconfirmed answer, and it is cached | `tavily_search.py:283-286, 711-716` | Groundedness; conflicts with E2 | Medium | codex, agy |
| F21 | The corpus loader exits the process on a missing or empty folder, reads only `.md`/`.txt`, skips unreadable files silently, and keys files by basename (duplicates collide) | `src/doc_chat/corpus.py:28-60` | Ingestion; D3 | Medium for RAG | codex, agy |

Smaller items still open from the last review: CR-2 (corrected lookup), CR-3 ("turn it up" after a tool-less turn), R4 list (`grounded_research` query, publisher names, untested retry-skip path) (`.scratch/context-p1-p3/REVIEW.md`).

## 8. RAG readiness against `docs/SEARCH-SPEC.md`

| Req. | Status | Evidence |
|---|---|---|
| E1 every claim traces to evidence | partial | web: source lead-in and backstop; no claim-level support (F1); docs: only on the race path (F4) |
| E2 abstain when evidence is missing | done for changeable web facts | `HEDGE`, `tavily_search.py:719-720` |
| E3 source leads the claim | done for web answers | `attribute()`; backstop |
| E4 documents carry dates | missing | no metadata (`docs.py`) |
| E5 conflicts reported with both sources | partial (web second look notice only) | `agent_server.py:631-636` |
| E6 file/web text is data | partial | research wrapper; docs unwrapped |
| E7 a failed lookup is a failure | done (S2) | `agent_tools.py:1060-1075` |
| E8 no memory fallback for changeable | partial | MCP side done; pipeline note misses hedged turns (F2) |
| S1, S2, S4, S7 | done | spec section 3 and code above |
| S3 latency budget | met at the median (ledger 1.73 s) | |
| S5 "the documents don't say" + web offer | missing (race falls through to web silently) | `agent_server.py:389-399` |
| S6 one clarifying question | prompt only | `system_agent.txt:24, 26` |
| S8 "reportedly" | prompt only, unmeasured | `system_agent.txt:34` |
| S9 file and date in a document answer | missing | |
| S10, S11 token-support check | missing | |
| S12, S13 second look on research | missing | `web_verify` only on `turn_claim()` |
| D1-D8 ingestion, change detection, failures by name, quoted answers with dates, doc+web pairing, conflicts, untrusted wrapping, time scope | missing (D4's quote contract exists on the race path only) | |

## 9. What the readings say, mapped to this code

| Principle | Source | Here today | Action |
|---|---|---|---|
| Observe → benchmark → evaluate → improve; evaluate before and after every change | **[EV-M1 §2-§3]**, **[AAI-M4 §7, §16]**, **[HARN L6]** | changes are verified by single live sessions and unit tests | build the loop: traces, a fixed benchmark, multi-run pass rates |
| Traces with spans, sessions | **[EV-M1 §6]**, **[CE-M1 §7]**, **[BOOK p.231-234]** | flat turn rows | span per call with racer, args, result hash, sources, latency, tokens; session id |
| Generalise from a concrete failure into a requirement | **[EV-M2 §1]** | rules per incident (F11) | rewrite the prompt's rules as the spec's E/S requirements, each with a test |
| Stratified benchmark, ≥ 50 % hard, provenance tags | **[EV-M2 §2, §4-§5]** | 8 + 23 + 105 cases, mostly happy paths | the spec's case matrix + FreshQA false premises + regressions from logs |
| Code evaluators first; single-responsibility judges; kappa ≥ 0.7 on a held-out set; no 1-10 scales | **[EV-M3 §1-§4]**, **[AAI-M4 §2-§6]**, **[BOOK p.164-168]** | none for answers | code: search ran, token support (S10); judges: relevance, abstention, binary, validated |
| Process vs outcome; required presence / forbidden absence | **[EV-M4 §1, §4-§5]** | process only | report both separately; never a composite score |
| Faithfulness vs answer relevance vs context utilisation; faithful-but-stale | **[BOOK p.160-161]**, **[RAG-M4 §7]** | none | three separate evaluators; stale-wins metric from the spec |
| Hallucination detection: LLM judge (2-5 s) or a small classifier (HHEM, 0-1 score) | **[BOOK p.92-95]** | none | offline: a judge in the benchmark; online: a small NLI classifier only if its latency fits (EverGreen precedent: 4 ms) **unverified for HHEM** |
| Post-tool hooks for currency, not prompt rules | **[CE-M1 §6, Lab 3 `check_current`]**, **[EV-M5 §3 layer 3]** | the repo already prefers code (gate, hedge, backstop) | a hook that labels each document passage with its date and flags undated/stale before the model sees it |
| Instruction defence: delimiters, data not commands | **[BOOK p.86-88]** | research wrapper only | wrap every passage; test with a hostile file (D7) |
| Versioned ingestion, entity id + version, delete old chunks; incremental refresh | **[BOOK p.73-74, p.162]**, **[RAG-M3 §3]** | static corpus | D1-D2 with hash and mtime; one version per file |
| Optimise retrieval before generation; measure Recall@k/Precision@k/MRR on golden chunks | **[BOOK p.158-160, p.168-174]**, **[RAG-M2 §7]**, **[CE-M2 §6]** | no retrieval metric | golden passages for the seeded corpus; then decide on hybrid |
| Hybrid BM25 + dense with RRF; two-stage rerank; contextual chunk headers; query rewriting / expansion | **[RAG-M2 §6]**, **[RAG-M3 §4-§7]**, **[BOOK p.75-82]**, **[ARET L3-L4]** | BM25 only | only if Recall@k is short; contextual headers (file, date, heading) are cheap and serve S9 now |
| Tool descriptions are the interface; overlapping tools confuse | **[CE-M2 §4]**, **[CE-M1 §4]** | `web_answer` / `web_research` / `search_docs` overlap; the redirect patches it | sharpen descriptions with when-not-to-use and limits; measure selection in the dry run |
| Memory: only store what helps a future session; facts / rules / artefacts; two-phase write with provenance; versioning | **[CE-M4 §1-§5]**, **[BOOK p.227-229]** | none | P6 of the critique: episodes with a verdict, user-direct provenance first, versioned facts; a changeable fact never answered from memory |
| Core vs archival memory | **[LETTA Lab 3-5]** | profile only | profile = core block; episodes and ingested files = archival, searched on demand |
| Reflection with external feedback beats self-critique | **[AAI-M2 §3, §10-§11]**, **[HARN L4]** | the second look is external feedback (good) | extend to research answers (S12) |
| One change at a time; dual suite (capability + known-good); pre-set acceptance bars | **[EV-M5 §1-§2]**, **[HARN L6]** | ad hoc | each prompt or gate change runs the whole benchmark before it lands |
| Latency and cost per step | **[AAI-M4 §14-§15]**, **[BOOK p.109-115]** | timings yes, tokens no | usage metrics on; count duplicate lookups |

## 10. What not to do

- **No vector database and no embeddings yet.** BM25 over a few hundred chunks is fast and exact on names; add dense retrieval only if measured Recall@k on golden passages falls short **[BOOK p.158]**, **[RAG-M2 §6]**. The Pi has no model runtime but Silero (`gate.py:8-9`).
- **No LLM judge on the live path for short answers.** A judge costs 2-5 s **[BOOK p.93]**; spoken answers are one or two sentences. Use code checks live (S10) and judges offline.
- **No compaction or summarisation of a 4-pair window** **[CE-M3 §4]** (nothing to compact; summaries are lossy, §5).
- **No more prompt rules without a benchmark run** **[EV-M5 §4]**.
- **No composite score** **[EV-M4 §5]**, **[EV-M3 §5]**.

## 11. Open questions for the owner

1. File formats and drop point for ingestion (spec §9).
2. Approve the grounding fixture capture (about $0.01) to close CR-1.
3. Keep the hedge race for lookup turns, or let the router's committed lookup end Gemini's request?
4. Should a document answer to a changeable question always wait for the web second look (D5), at about +1-2 s?
5. Targets in spec §6 (unsupported-claim ≤ 2 %, abstention recall ≥ 95 %) as acceptance bars for the benchmark.

## 12. Roadmap (each step has an automated acceptance test)

**Status (2026-11-18, in code, uncommitted):** steps 1-3 are done (trace: `docs/TURN-TRACE.md`; benchmark v1 and evaluators: `docs/AGENT-BENCHMARK.md`). F2 is half closed by F2a (the changeable note now reaches hedged turns; its live effect is unmeasured); F2b and F19 are open.

**Step 5 status (2026-11-18, in code, uncommitted; `docs/RAG.md`):** ingestion D1-D3 and D7 are done (`DocIndex` with per-file and per-chunk metadata, change detection on every question, failures named, untrusted wrapping). D4 is done on the new `docs` route (cited answer with a verbatim quote, file, section or page and date) with a known limit: the quote check proves a quote exists, not that it answers. S5 and S9 are done on that route and partial elsewhere. D5, D6 and D8 are open (no doc + web pairing, conflict reporting or time scope). Step 7's retrieval metrics exist (`python -m voice_agent.rag.metrics`, golden set of 29 answerable questions) and a hybrid retriever (BM25 + answerai-colbert-small-v1) is built, default `auto` with BM25 fallback. F4 is partial: the docs route runs the quote and names/numbers checks and the model route's `search_docs` passages are wrapped as untrusted, but a Case C answer the model writes itself still has no quote check. F5 is partial: dates, hashes and change detection exist; versions do not, and the documents-vs-web race in `doc_race` is unchanged.

| Step | Work | Acceptance test |
|---|---|---|
| 1 | Trace per turn: session id, span per call (racer, tool, args, result hash, sources, latency), tokens (`usage=True`); second-look rows to their own file; turn id in the ledger | a replayed turn produces spans whose racer and tool match the calls; tokens > 0 |
| 2 | Benchmark v1: spec case matrix (12) + FreshQA false-premise sample + 20 regressions from logs + happy paths; ≥ 50 % hard; scripted multi-turn | runs offline with fakes in CI; live run reports per-evaluator pass rates over 5 runs |
| 3 | Evaluators: process (lookup, searched), support (S10/S11 tokens), relevance (binary judge, kappa ≥ 0.7 on 25 % held out), abstention, source coverage | each evaluator has its own fixture tests; no composite |
| 4 | Close F1-F3, F6, F7, F13-F20 one at a time. Decide on codex's **release gate**: factual text is held until its evidence record exists (the `say` progress sentence still plays at once), so an unsupported answer is blocked rather than logged as `unchecked` (`agent_server.py:741-744, 875-884`; `pipeline/tests/test_changeable.py` today asserts logging, not blocking) | per route (fast, router, hedged Gemini, Case C, doc race, stall fallback), injected no-lookup / zero-search / failed / cancelled results give 0 unsupported facts released; first-text latency reported before and after |
| 5 | Ingestion D1-D3, D7 with contextual chunk headers; Case C gets the doc-answer checks (F4) | seeded corpus: every chunk has file, mtime, hash; a hostile file changes nothing; Case C answers pass the quote check |
| 6 | D4-D6, D8, S5, S9; doc + web pairing for changeable questions | stale-wins 0 on the seeded set; conflicts name both files and dates |
| 7 | Retrieval metrics on golden passages; hybrid or rerank only if Recall@3 < target | Recall@k and MRR reported; latency within S3 |
| 8 | Memory P4-P6 | referring-back fixtures; a changeable fact from memory still triggers a search |

## 13. Comparison with the other reviewers

Both were run on 2026-11-18 from the same brief (`docs/agent-assessment/BRIEF.md`, launched by `.scratch/agent-assessment/run.sh`), read-only, without this document's findings: agy with `gemini-4-argon-high` (`docs/agent-assessment/AGY-ARGON.md`, 150 KB) and codex with `gpt-6.1-terra`, reasoning xhigh (`docs/agent-assessment/CODEX-TERRA.md`, 56 KB, 210k tokens). Every claim was checked against the code, logs or ledgers, three by running the code: **`docs/agent-assessment/VALIDATION.md`** (codex 17/17 confirmed; agy 17 confirmed, 4 partly, 12 wrong). F13-F21 in section 7 came from them.

### 13.1 Where all three agree

- The changeable note does not reach hedged turns, and the pipeline uses the regex gate while the lookup uses EverGreen (F2).
- Case C document answers bypass the quote check, and passages are not wrapped as untrusted (F4, F12).
- No ingestion lifecycle: no dates, hashes, versions, change detection (F5, F21; D1-D8 missing).
- Attribution is a label, not a claim-to-evidence link; the server backstop is trailing (sections 3.3, 8).
- No traces, no tokens, no racer attribution (F8); build the benchmark and traces before changing retrieval; no vector database yet.

### 13.2 What each found that the others missed

| Reviewer | Unique, verified findings |
|---|---|
| This document | the hedge race duplicating lookups in 40/54 turns (F3); the answer-relevance failure in the Nvidia turn; the zero-search retry gaps in the ledger; the mapping of every reading to a code location |
| codex | the release-gate framing (an unsupported answer is logged, not blocked); the negation counterexample in `doc_answer` (F16); the cache dropping tense words (F15); the correction notice missing from history (F18); `passages_of` dropping the file name; the gate-outcome rows that were SUPPORTED with zero searches; dry-run scoring collapsing repeated calls (`dryrun.py:166`, not re-checked here) |
| agy | the stale chain crossing a restart (F13, the most serious new finding); the failure text propagating through the shared lookup future (F14); `web_research` results never getting a backstop source and the "source" substring suppression (F17); `^question-word` in `changeable_question` (F19); digit-substring and comma mismatches in `doc_answer` (F16); `corpus.py` exiting on an empty folder (F21) |

### 13.3 Where agy was wrong (checked, not adopted)

agy's report is long and specific, but several specifics are invented or misattributed. Do not act on its text without checking:

| Claim | Fact |
|---|---|
| The losing hedged request runs `chain_update` too and overwrote the chain back to KoolPals | `gated_tools` runs a handler only for the winner and returns `superseded` to the loser (`hedge.py:65-92`); the log shows `chain_update` from `GoogleLLMService#1` only (`agent.log:265-266, 305-306`). The stale chain is real (F13); the mechanism given is not |
| `gate_outcomes.jsonl` line 4 is a DIFFERS verdict on a Joe Rogan question | line 4 is SUPPORTED on "Who won the last NBA championship?"; the file has no DIFFERS row |
| Ledger rows carry `changeable`, `grounded`, `queries`; 119 rows are `grounded: false`; a 7-day ledger TTL; line 1 is the swallow question | ledger rows hold `time, month, searches, tokens, seconds` (+ `query, sources, forced, check`); no such fields; the answer cache is 600 s (`tavily_search.py:74`) |
| `FAILURE` is `\b(could not|couldn't|failed|unable to|error|...)\b`; failed calls are rewritten to `"FAILED: <tool>"` | `FAILURE` is `couldn't find|did not answer|not connected|can't say|error` (`agent_tools.py:185`); there is no "FAILED:" rewrite |
| The client uses a UNIX socket with an HMAC challenge; the online probe is TCP to 8.8.8.8; the router is a 0.5B FunctionGemma; usage is OpenRouter's | WebSocket with a token header (`agent_server.py:116-132`); the probe is `generativelanguage.googleapis.com` (`:140`); the router is Granite 4.0 H 350M (`README.md`); the front model is Gemini through Pipecat |
| `run_bash` always enables network in the sandbox | the cited symbols (`allow_network`, `_build_bwrap_cmd`) do not exist; the runner shares the network only when `policy.network` is set (`src/voice_agent/sandbox/runner.py:32, 185-186`). **Unverified either way** which policy `run_bash` uses |
| Reading paths such as `RAG/source-notes/M3_advanced-rag-techniques.md`, `M4_domain-specific-rag-and-trusted-generation.md` | those files do not exist (the real names are in section 14) |

### 13.4 How the outputs differ

| | This document | codex (gpt-6.1-terra xhigh) | agy (gemini-4-argon-high) |
|---|---|---|---|
| Style | tables, short rows, code and reading keys | prose with precise hedging ("inferred, not tested", "unverified"), every claim cited | very long, mermaid diagram, severity × frequency scores, exclamation marks, file:// links |
| Accuracy | verified against logs; one stale count corrected after codex's read (gate outcomes) | no false claim found in the checks made here | several fabricated specifics (13.3) next to real, deep findings |
| Use of logs | quantitative (counts, medians, joins) | cautious, descriptive | reads raw agent.log; best at reconstructing an incident, worst at quoting data |
| Recommendations | an ordered roadmap with acceptance tests | 7 contract-first recommendations with numeric acceptance tests | 9 recommendations with code sketches, P0-P2 |
| Use for this project | the master plan | the second opinion to trust on evidence contracts | a lead generator: every claim needs checking |

## 14. Sources

**Readings** (second-brain notes under `/mnt/Obsidian/Claude/second-brain/resources/`; each folder has `NOTES.md` and `source-notes/M*.md`):

| Key | Source |
|---|---|
| EV-M1..M5 | `evaluating-and-improving-agentic-applications/source-notes/M1_why-evals-matter-and-traces.md` ... `M5_improving-agentic-applications.md` (DeepLearning.AI & Arize) |
| RAG-M1..M5 | `RAG/source-notes/M1_intro-to-rag.md` ... `M5_rag-systems-in-production.md` (DeepLearning.AI & Weaviate) |
| CE-M1..M4 | `context-engineering/source-notes/M1_intro-context-engineering.md` ... `M4_storing-context.md` (DeepLearning.AI & Chroma) |
| AAI-M1..M5 | `agentic-ai/source-notes/M1_intro-agentic-workflows.md` ... `M5_planning-multi-agent.md` (DeepLearning.AI, Andrew Ng) |
| LETTA Lab 3-6 | `/mnt/Others/learning/extra-resources/Building_agents_with_Letta.html` (Lab 3, core and archival memory), `Customizing_memory_management_in_MemGPT.html` (Lab 4, memory blocks, task queue), `Agentic_RAG_and_Accessing_External_Memory.html` (Lab 5, sources and custom tools), `Orchestrating_Agents_with_MemGPT.html` (Lab 6, shared blocks, multi-agent) |
| HARN L3, L4, L6 | `extra-resources/L3.html` (agent loop, AGENTS.md, MAX_TURNS), `L4.html` (verification loop: rubric grader, iteration cap), `L6.html` (hill-climbing: batch, recurring gaps, one change, compare) |
| ARET L3-L5 | `extra-resources/L3-student.html` (query expansion, generated answers / multiple queries), `L4-student.html` (cross-encoder reranking), `L5-student.html` (embedding adaptors) |
| FT L5 | `extra-resources/Lesson5.html` (generating and validating synthetic SQL training data; relevant to the router's distillation data) |
| BOOK | Mendelevitch & Bao, *Hands-On RAG for Production*, `extra-resources/Hands-On RAG for Production ... .pdf`: index freshness p.64, document updates p.73-74, two-stage retrieval p.75, hybrid p.76, rerank p.79, prompt injection p.86-88, hallucinations p.88-95, UX p.96, latency p.109, failure modes p.157-163, LLM-as-judge p.164, metrics p.168-177, agentic memory p.227-229, observability p.229-234 |

**Repository documents**: `docs/SEARCH-SPEC.md`, `docs/CONTEXT-CRITIQUE.md`, `docs/TEXT-MODE.md`, `docs/MCP-SERVERS.md`, `docs/CITATION-TAXONOMY.md`, `.scratch/context-p1-p3/{PLAN,DESIGN,REVIEW}.md`.
