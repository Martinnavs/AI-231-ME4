> **Verbatim output** of codex `gpt-6.1-terra` (reasoning xhigh), read-only, 2026-11-18 06:48-06:53 (210,463 tokens), from `BRIEF.md`. Not edited. **Validation: `VALIDATION.md` §2** — every claim checked was correct; claims not checked are listed there. Adopted findings are F14-F16, F18, F20-F21 in `../AGENT-ASSESSMENT.md`.

---

# Independent assessment of the voice/text agent

## Assessment and scope

The agent has useful latency engineering and several targeted safeguards, but it is **not yet ready to guarantee the evidence contract in `docs/SEARCH-SPEC.md`**. The most consequential gap is at the final answer boundary: the server can stream a factual answer before determining whether a lookup occurred, and its current post-turn flag is observational rather than preventive. The document path has a second gap: its quote and token checks are useful heuristics, but they do not establish that the answer follows from the quote, preserve the file identity through the document race, or establish freshness (`pipeline/src/voice_pipeline/agent_server.py:741`, `pipeline/src/voice_pipeline/agent_server.py:868`, `pipeline/src/voice_pipeline/doc_answer.py:53`, `pipeline/src/voice_pipeline/doc_answer.py:109`).

The shortest path to RAG readiness is to establish a structured evidence contract across **all** answer paths, enforce it before factual text is emitted, and build a small labelled evaluation set before changing retrieval technology. That work also benefits today’s web search.

This assessment is read-only. I did not run the agent, execute tests, make network calls, access key or token files, or modify files. I inspected the specified code, design documents, local records, and relevant sections of the supplied readings. The readings are course notes and book extracts; I did not independently verify their underlying sources.

The local server records examined contain 54 turn rows, and the grounding ledger contains 418 rows. They span earlier runs and may reflect code revisions or configurations different from the current checkout. Their counts are descriptive, not production failure rates. The checkout is on `text-only`; an existing untracked `.scratch/agent-assessment/` directory was present before this review.

**Reading references below** use these roots:

- `evaluation/` = `/mnt/Obsidian/Claude/second-brain/resources/evaluating-and-improving-agentic-applications/`
- `RAG/`, `context-engineering/`, and `agentic-ai/` = their sibling directories under `/mnt/Obsidian/Claude/second-brain/resources/`
- `ragbook.txt` = the supplied scratchpad extraction of *Hands-On RAG for Production*.

## 1. How one turn works

### Request, routing, and model calls

1. The text client retains conversation history in memory and sends the last four exchanges by default with the new user message. It displays streamed text and tool names; background notices are displayed before a later prompt (`src/text_chat/cli.py:34`, `src/text_chat/cli.py:47`, `src/text_chat/cli.py:62`, `src/text_chat/cli.py:73`, `src/text_chat/client.py:73`).

2. `AgentServer.handle` discards client-supplied system messages, serializes turns with a lock, sets the current question, resets per-turn lookup state, and chooses the execution path (`pipeline/src/voice_pipeline/agent_server.py:723`, `pipeline/src/voice_pipeline/agent_server.py:757`, `pipeline/src/voice_pipeline/agent_server.py:763`).

3. Backchannels and standalone apologies can receive no answer. Recognized fixed commands and clock questions use the fast path. An active research/build chain normally sends the turn to Gemini with chain state and the Case F protocol. Otherwise the router competes with Gemini, subject to exceptions such as a reply to an offer (`pipeline/src/voice_pipeline/agent_server.py:796`, `pipeline/src/voice_pipeline/agent_server.py:810`, `pipeline/src/voice_pipeline/agent_server.py:813`, `pipeline/src/voice_pipeline/agent_server.py:821`).

4. The local router gets the current utterance and at most one prior user/router pair. It emits grammar-constrained calls or an escalation; policy checks can reject its result and pass the turn to Gemini (`pipeline/src/voice_pipeline/router.py:27`, `pipeline/src/voice_pipeline/router.py:97`, `pipeline/src/voice_pipeline/router.py:129`, `pipeline/src/voice_pipeline/agent_server.py:225`).

5. In the normal text-stack configuration, two Gemini requests start immediately alongside the router. The first participant to claim the turn through text or a tool call wins. Tool handlers are gated so the losing model should not perform side effects; losing requests are cancelled. The runtime’s default router hold is zero, so a model can win before the router completes (`scripts/stack/up.sh:152`, `pipeline/src/voice_pipeline/agent_server.py:913`, `pipeline/src/voice_pipeline/agent_server.py:983`, `pipeline/src/voice_pipeline/hedge.py:57`, `pipeline/src/voice_pipeline/hedge.py:65`, `pipeline/src/voice_pipeline/hedge.py:138`).

6. On an eligible routed web lookup, the server can start a web lookup and a document answer concurrently. It waits for the document check first; if accepted, the document answer wins and the web task is cancelled. Otherwise it uses the web result if available. This is document preference with a timeout, not a comparison of the two sources (`pipeline/src/voice_pipeline/agent_server.py:336`, `pipeline/src/voice_pipeline/agent_server.py:364`, `pipeline/src/voice_pipeline/agent_server.py:382`, `pipeline/src/voice_pipeline/agent_server.py:457`).

7. Responses pass through streaming payload, source-aside, and sentence filters. The server may append a source sentence after the answer, sends `done`, schedules a background second look when it has a recognizable lookup claim, logs the turn, and updates chain state (`pipeline/src/voice_pipeline/agent_server.py:732`, `pipeline/src/voice_pipeline/agent_server.py:841`, `pipeline/src/voice_pipeline/agent_server.py:851`, `pipeline/src/voice_pipeline/agent_server.py:868`, `pipeline/src/voice_pipeline/agent_server.py:870`).

### What the model sees

- The main system prompt defines Cases A–G, tools, short spoken format, clarifying behavior, and source wording. At startup it can receive additional profile, owner, venue, and publishing instructions (`pipeline/config/eval/system_agent.txt:1`, `pipeline/config/eval/system_agent.txt:14`, `pipeline/config/eval/system_agent.txt:26`, `pipeline/config/eval/system_agent.txt:30`, `pipeline/src/voice_pipeline/agent_server.py:943`).
- The tool schemas come from the connected MCP servers and local wrappers, with some tools hidden from the front model. The available set is configuration-dependent (`pipeline/src/voice_pipeline/agent_tools.py:823`, `pipeline/src/voice_pipeline/agent_tools.py:972`, `pipeline/src/voice_pipeline/agent_server.py:963`).
- A normal steered call can append active chain state, the Case F protocol, a regex-based changeability note, and a current date/time/place line to the last user message (`pipeline/src/voice_pipeline/agent_server.py:241`, `pipeline/src/voice_pipeline/agent_server.py:251`, `pipeline/src/voice_pipeline/agent_server.py:900`).
- **Important exception:** when the router and hedged models start together, the model requests receive only the timestamp suffix, because they launch before the router’s decision. They do not receive `steer`’s changeability note or Case F tail on that path (`pipeline/src/voice_pipeline/agent_server.py:264`, `pipeline/src/voice_pipeline/agent_server.py:587`). The base prompt still contains general instructions.
- The text client sends recent spoken text, not prior tool arguments, retrieved passages, or structured evidence. Gemini’s request replaces its conversation context with the supplied messages on each turn (`src/text_chat/cli.py:34`, `src/text_chat/cli.py:80`, `pipeline/src/voice_pipeline/agent.py:405`).

### Memory

- **Within the client session:** recent user/assistant text is replayed; the default is four exchanges. The router gets only one prior pair, and a Gemini-written answer is shortened for that router history (`src/text_chat/cli.py:34`, `pipeline/src/voice_pipeline/router.py:29`, `pipeline/src/voice_pipeline/agent_server.py:225`).
- **Task state:** `chain.json` stores phase, subject, model-supplied facts, items, job and artifact information. It can survive a server restart while still active, but expires after ten minutes of inactivity or three unrelated turns. It is task state, not verified long-term memory (`pipeline/src/voice_pipeline/chain_state.py:26`, `pipeline/src/voice_pipeline/chain_state.py:140`, `pipeline/src/voice_pipeline/chain_state.py:164`, `pipeline/src/voice_pipeline/chain_state.py:243`).
- **Delegate scratchpad:** recent lookup results and chain facts may be written to a 12 KB, ten-minute scratchpad labelled as untrusted and made available to the delegate. The main front model does not automatically receive the scratchpad contents (`pipeline/src/voice_pipeline/scratchpad.py:18`, `pipeline/src/voice_pipeline/scratchpad.py:80`, `pipeline/src/voice_pipeline/agent_tools.py:1085`, `pipeline/src/voice_pipeline/agent_tools.py:1144`).
- **Other retained state:** process caches, logs, ledger records, a filled profile, and the verification allowance persist in different ways. They should not be described collectively as conversational memory (`src/voice_agent/mcp_servers/tavily_search.py:74`, `pipeline/src/voice_pipeline/agent_server.py:943`, `src/voice_agent/mcp_servers/gemini_grounding.py:289`).
- A correction notice is added to the text UI and acknowledged, but the notice-handling loop does not add it to the `history` list sent on the next turn. **Inferred, not tested:** a subsequent model request can therefore see the earlier answer without seeing its correction (`src/text_chat/cli.py:62`, `src/text_chat/cli.py:64`, `src/text_chat/cli.py:80`).

## 2. Inventory of checks and guards

| Check or guard | What it does | Limit |
|---|---|---|
| **Fast-path matching** | Recognizes fixed home commands and clock questions, invokes a tool or local clock, and formats a result (`pipeline/src/voice_pipeline/fast_path.py:240`, `pipeline/src/voice_pipeline/agent_server.py:197`). | It is a pattern-based shortcut, not a general intent or factuality check. |
| **Router grammar and deadline** | Restricts the router’s output format; parse errors, unavailable tools, escalation, and timeout fall through to Gemini (`pipeline/src/voice_pipeline/router.py:27`, `pipeline/src/voice_pipeline/router.py:97`). | A syntactically valid call can still be semantically wrong. |
| **Router call policy** | Checks, among other things, unsupported intents, numeric slots against the user’s words, negation, keywords, place validity, document topic, and whether a lookup was requested (`pipeline/src/voice_pipeline/call_policy.py:490`, `pipeline/src/voice_pipeline/call_policy.py:534`, `pipeline/src/voice_pipeline/call_policy.py:552`). | These checks apply to the routed path; they do not prove the correctness of the retrieved answer or enforce the same checks on every model path. |
| **Hedge gate** | The winner claims the turn before a model tool handler runs; losing handlers return `superseded` (`pipeline/src/voice_pipeline/hedge.py:39`, `pipeline/src/voice_pipeline/hedge.py:65`). | It chooses the first signal, not the most supported or complete answer. Cancellation cannot undo an action already performed by the winner. |
| **Front changeability note and audit** | A regex can append a lookup instruction and, after the turn, log `unchecked` if no recognized lookup or intent appears (`pipeline/src/voice_pipeline/call_policy.py:643`, `pipeline/src/voice_pipeline/agent_server.py:256`, `pipeline/src/voice_pipeline/agent_server.py:875`). | It is not a release gate. The speculative hedged model path receives only the timestamp. The audit counts a recorded call or intent, not a successful search (`pipeline/src/voice_pipeline/agent_server.py:587`, `pipeline/src/voice_pipeline/agent_server.py:877`). |
| **MCP changeability classifier** | The web tool uses the local EverGreen classifier where available, falling back to a regex when it cannot load or classify (`src/voice_agent/gate.py:24`, `src/voice_agent/gate.py:63`, `src/voice_agent/mcp_servers/gemini_grounding.py:139`). | It operates inside the web tool, after the front agent has chosen to call that tool. A classification miss can accept a zero-search answer (`src/voice_agent/mcp_servers/tavily_search.py:569`). |
| **Grounded search and retry** | For a tool-classified changeable query, a zero-search Gemini answer triggers a forced retry when time and budget permit; if still unsearched, the path tries Tavily (`src/voice_agent/mcp_servers/tavily_search.py:558`). | Search-query count proves that a search was issued, not that a source supports every claim. A query considered settled can return a zero-search model answer (`src/voice_agent/mcp_servers/tavily_search.py:572`, `src/voice_agent/mcp_servers/tavily_search.py:581`). |
| **Tavily and Wikipedia fallbacks** | Tavily supplies a short generated answer or top snippet. The code checks whether query names and numbers appear in the answer or snippets; Wikipedia is the final fallback in some failure cases (`src/voice_agent/mcp_servers/tavily_search.py:230`, `src/voice_agent/mcp_servers/tavily_search.py:256`, `src/voice_agent/mcp_servers/tavily_search.py:686`, `src/voice_agent/mcp_servers/web_info.py:203`). | Query-token presence is not answer entailment. On missing query tokens, the Tavily path still speaks the “closest result.” Wikipedia returns an article extract rather than a claim-level answer (`src/voice_agent/mcp_servers/tavily_search.py:283`, `src/voice_agent/mcp_servers/tavily_search.py:709`). |
| **No-guess hedge** | If a changeable Gemini answer was unsearched and Tavily cannot check it, the explicit hedge contains no guess (`src/voice_agent/mcp_servers/gemini_grounding.py:321`, `src/voice_agent/mcp_servers/tavily_search.py:719`). | That guarantee is narrow: other paths can still deliver a closest result or an unsupported answer. |
| **Per-turn lookup deduplication** | A second lookup can wait for the first or reuse its result. If a cancelled first call leaves an empty result, the second call proceeds instead of receiving an empty “already looked up” (`pipeline/src/voice_pipeline/agent_tools.py:1054`). | Reuse is not keyed by the requested query or tool. Failure classification is based on a short regex; nonempty failures outside it can be treated as reusable results (`pipeline/src/voice_pipeline/agent_tools.py:185`, `pipeline/src/voice_pipeline/agent_tools.py:1056`, `pipeline/src/voice_pipeline/agent_tools.py:1085`). |
| **Cross-turn answer cache** | Similar sets of content words can share a cached answer for ten minutes, and in-flight equivalent requests are joined (`src/voice_agent/mcp_servers/tavily_search.py:74`, `src/voice_agent/mcp_servers/tavily_search.py:243`, `src/voice_agent/mcp_servers/tavily_search.py:416`). | The stop-word and overlap rules can erase distinctions such as present versus past; the cache is not keyed by a structured temporal scope (`src/voice_agent/mcp_servers/tavily_search.py:87`). |
| **Attribution** | Grounding metadata yields publisher names where possible; the web answer can lead with “As per,” and the server can append a source sentence if the model omitted one (`src/voice_agent/mcp_servers/gemini_grounding.py:111`, `src/voice_agent/mcp_servers/gemini_grounding.py:207`, `src/voice_agent/mcp_servers/gemini_grounding.py:238`, `pipeline/src/voice_pipeline/agent_tools.py:382`). | The backstop is after the claim, applies only to recognized source text, and does not associate each claim with an evidence span. `attribute` returns unmodified text when no usable source name exists (`src/voice_agent/mcp_servers/gemini_grounding.py:238`). |
| **Second look** | After a recognizable web lookup claim, a budgeted background call forces search and compares the first lookup claim with a second answer. Current code requires the second call to have searched before returning `SUPPORTED` or `DIFFERS`; differences can become notices (`pipeline/src/voice_pipeline/agent_tools.py:406`, `src/voice_agent/mcp_servers/tavily_search.py:632`, `pipeline/src/voice_pipeline/agent_server.py:612`). | It happens after the answer; compares the lookup claim rather than necessarily the final reworded answer; can be skipped, fail, or exhaust its allowance; and is not a claim-by-claim proof (`pipeline/src/voice_agent/mcp_servers/tavily_search.py:633`, `pipeline/src/voice_pipeline/agent_server.py:625`). |
| **Document retrieval and answer check** | Startup loads Markdown/text files into 700-character overlapping BM25 chunks; search returns source basename and text. The special document-answer path asks for an answer and quote, then checks normalized quote occurrence and answer names/numbers (`src/voice_agent/mcp_servers/docs.py:24`, `src/voice_agent/mcp_servers/docs.py:27`, `pipeline/src/voice_pipeline/doc_answer.py:19`, `pipeline/src/voice_pipeline/doc_answer.py:109`). | The quote check is normalized, not exact byte-for-byte; it does not prove entailment. `passages_of` discards source identity and the accepted result returns the answer alone. The regular Case C prompt can also call `search_docs` directly and answer from the passages without going through this special checker (`pipeline/src/voice_pipeline/doc_answer.py:39`, `pipeline/src/voice_pipeline/doc_answer.py:75`, `pipeline/src/voice_pipeline/doc_answer.py:120`, `pipeline/config/eval/system_agent.txt:20`). |
| **Untrusted web data** | Tavily research strips markup and a small set of instruction-like strings and wraps search results in an untrusted-data tag (`src/voice_agent/mcp_servers/tavily_search.py:180`, `src/voice_agent/mcp_servers/tavily_search.py:196`). | A finite string filter is not a security boundary. Document passages are inserted into the document-answer request without an equivalent explicit untrusted-data wrapper (`pipeline/src/voice_pipeline/doc_answer.py:95`). |
| **Output filters** | JSON-looking objects and parenthetical source asides are suppressed; status sentences can be deduplicated and certain premature job-completion claims dropped (`pipeline/src/voice_pipeline/payload_guard.py:11`, `pipeline/src/voice_pipeline/payload_guard.py:62`, `pipeline/src/voice_pipeline/spoken_filter.py:88`). | These are presentation checks, not factual support checks. The payload filter explicitly does not fully parse JSON strings (`pipeline/src/voice_pipeline/payload_guard.py:11`). |
| **Tool/agent stall and network recovery** | Routed read-only lookups can retry transient failures; the model has first-chunk and turn deadlines, plus a stall fallback that may speak a usable tool result or apologize (`pipeline/src/voice_pipeline/agent_server.py:329`, `pipeline/src/voice_pipeline/agent.py:53`, `pipeline/src/voice_pipeline/agent.py:134`). | A fallback improves availability but does not establish claim support. |
| **Delegation and action guards** | Destructive delegate requests and publishing have confirmation logic; job starts receive fixed acknowledgements; task facts and scratchpad material may be passed to the delegate (`pipeline/src/voice_pipeline/agent_tools.py:1115`, `pipeline/src/voice_pipeline/agent_tools.py:1183`, `pipeline/src/voice_pipeline/agent_tools.py:1212`). | Regex confirmation and task-memory plumbing do not establish that research facts are correct. These are separate from the answer-evidence contract. |

## 3. Problems and risks, ranked

The ranking combines potential harm with observed exposure. Frequency estimates beyond the cited local samples are **unverified**. Taxonomy labels follow the five agent failure modes in `evaluation/NOTES.md §Settled Terminology & Taxonomy`, the four context failure modes in `context-engineering/source-notes/M1_intro-context-engineering.md §4`, and retrieval, generation, and ingestion failures in `ragbook.txt ch. 6, “How Does RAG Fail?”`.

### 1. High severity; exposure depends on routing: factual output can escape before evidence exists

The front server streams text as it arrives and only later adds an `unchecked` log flag. A changeable question can therefore be answered without a lookup; a tool call in the log also does not establish that the lookup succeeded. The hedged path does not receive the per-turn changeability note (`pipeline/src/voice_pipeline/agent_server.py:741`, `pipeline/src/voice_pipeline/agent_server.py:587`, `pipeline/src/voice_pipeline/agent_server.py:875`). The existing test deliberately verifies that an answer with no lookup is *logged* as unchecked, not blocked (`pipeline/tests/test_changeable.py:100`).

**Classification:** execution, groundedness, and potentially propagation; context poisoning if an unsupported answer becomes later context; RAG generation faithfulness failure. Actual frequency on current traffic is **unverified**.

### 2. High severity; frequent by construction on document answers: quote matching is weaker than support

The document checker accepts a normalized quote anywhere in the joined passages and checks selected names and numbers. It does not establish that the quote supports the answer, that the answer preserves negation, or that the answer addresses the question. It ignores the first answer word for the capitalized-name check and does not enforce the prompt’s requested quote length (`pipeline/src/voice_pipeline/doc_answer.py:53`, `pipeline/src/voice_pipeline/doc_answer.py:60`, `pipeline/src/voice_pipeline/doc_answer.py:109`).

**Static counterexample, unverified by execution:** given the passage and quote “Alice is the CEO,” an answer “Alice is not the CEO” appears able to pass: the quote occurs and the remaining capitalized token `CEO` occurs. This is a generation-faithfulness failure even though a quote is present.

The same mechanism drops the source metadata before judging and returns only the answer. The direct Case C path does not use this checker at all (`pipeline/src/voice_pipeline/doc_answer.py:75`, `pipeline/src/voice_pipeline/doc_answer.py:120`, `pipeline/config/eval/system_agent.txt:20`).

**Classification:** judgment and groundedness; context poisoning; RAG generation faithfulness and answer relevance.

### 3. High severity; likely on changing or conflicting material: sources, dates, and conflicts are not bound to claims

A source label is not an evidence span. The grounding parser selects an index associated with an early supported claim and can name up to two publishers, but the answer is not checked claim by claim. The server’s source backstop can append a generic source line after the answer. The document race can choose the document answer without comparing it with the concurrently obtained web answer (`src/voice_agent/mcp_servers/gemini_grounding.py:111`, `src/voice_agent/mcp_servers/gemini_grounding.py:207`, `pipeline/src/voice_pipeline/agent_tools.py:382`, `pipeline/src/voice_pipeline/agent_server.py:382`).

The older local logs illustrate why this matters without establishing which answer is true: two runs gave different “latest episode” answers, and another later run gave a third (`pipeline/out/agent_server/20261116-104540.jsonl:12`, `pipeline/out/agent_server/20261116-104540.jsonl:13`, `pipeline/out/agent_server/20261116-120321.jsonl:7`). A research answer also contained a named-person allegation without a source in the same sentence in one earlier run (`pipeline/out/agent_server/20261116-104540.jsonl:8`). The truth of those claims is **unverified** and is not needed to establish the attribution and consistency problems.

**Classification:** groundedness, judgment, propagation; context clash and poisoning; RAG ingestion staleness and generation/context-utilization failures.

### 4. High severity; frequency unmeasured: retrieval reuse can answer a different question

Within a turn, the first web lookup result can be returned for a later web lookup regardless of the later query or tool. Across turns, cache matching uses content-word overlap and removes question and temporal words including `who`, `was`, `current`, and `now` (`pipeline/src/voice_pipeline/agent_tools.py:1054`, `src/voice_agent/mcp_servers/tavily_search.py:87`, `src/voice_agent/mcp_servers/tavily_search.py:250`, `src/voice_agent/mcp_servers/tavily_search.py:416`).

**Inferred, not tested:** “Who is the CEO of Nvidia?” and “Who was the CEO of Nvidia?” can reduce to the same content words, allowing a present-tense cached answer to satisfy a historical question. This is particularly dangerous when a corrected question is paired with the prompt instruction to make no new lookup (`pipeline/config/eval/system_agent.txt:28`; the earlier review also identified that prompt issue at `.scratch/context-p1-p3/REVIEW.md:62`).

**Classification:** execution and propagation; context confusion and poisoning; RAG retrieval failure.

### 5. High severity for ingested files; currently systematic: no ingestion lifecycle or freshness model

The corpus loader only discovers `.md` and `.txt`, replaces invalid UTF-8, silently skips unreadable and empty individual files, and records only a basename. The retriever is constructed at server startup; no per-file hash, ingestion timestamp, content date, chunk ID, version, or query-time stale-index check is present in this path (`src/doc_chat/corpus.py:14`, `src/doc_chat/corpus.py:30`, `src/doc_chat/corpus.py:38`, `src/voice_agent/mcp_servers/docs.py:27`).

Two files with the same basename can become indistinguishable in the returned metadata. A newly indexed file can itself contain old claims: file modification time is useful provenance, but is not proof that the underlying fact is current.

**Classification:** execution and propagation; context poisoning and clash; RAG ingestion parsing and staleness, followed by retrieval or generation failures. The book explicitly distinguishes a faithful answer from an outdated source from an unfaithful answer (`ragbook.txt ch. 6, “Generation Failures” and “Failures Due to Inadequate Data Ingestion”`).

### 6. Medium-to-high severity; common blind spot: post-answer verification is incomplete and historically misleading

The current verifier requires a search, which corrects a historical defect. In the local outcome ledger, five earlier rows nevertheless report `SUPPORTED` with zero searches; three later zero-search rows report `UNVERIFIED`, and three later `SUPPORTED` rows show searches (`~/.local/state/voice_agent/gate_outcomes.jsonl:1`, `~/.local/state/voice_agent/gate_outcomes.jsonl:6`, `~/.local/state/voice_agent/gate_outcomes.jsonl:9`, `src/voice_agent/mcp_servers/tavily_search.py:645`).

The current check remains asynchronous, can be disabled or exhausted, and checks the tool’s stored claim rather than a complete final answer. A difference notice contains the second answer but does not itself ensure that both original and second sources and dates are presented (`pipeline/src/voice_pipeline/agent_tools.py:406`, `pipeline/src/voice_pipeline/agent_server.py:612`, `src/voice_agent/mcp_servers/tavily_search.py:632`).

**Classification:** groundedness, completion, propagation; context clash; RAG generation faithfulness. The ledger is historical evidence, not proof that current code still issues zero-search `SUPPORTED`.

### 7. Medium severity; frequency unmeasured: unsafe or irrelevant material can enter the model and task state

Web research has a labelled untrusted wrapper, but document passages are concatenated into the user content without an equivalent explicit boundary. Model-written chain facts can later be injected into delegate tasks as “unchecked” notes; the model is not required to attach evidence IDs to them (`src/voice_agent/mcp_servers/tavily_search.py:196`, `pipeline/src/voice_pipeline/doc_answer.py:95`, `pipeline/src/voice_pipeline/agent_tools.py:482`, `pipeline/src/voice_pipeline/agent_tools.py:1140`).

**Classification:** propagation and judgment; context poisoning and distraction; ingestion and generation failures. An actual successful prompt-injection exploit is **unverified**.

## 4. SEARCH-SPEC readiness

Status is assessed against the stated requirement, not merely the presence of a related helper. “Partial” includes prompt-only behavior, a narrow path, or an unmeasured guarantee.

### Evidence principles

| ID | Status | Evidence and gap |
|---|---|---|
| **E1** | **Partial** | Web source names and document token checks exist, but there is no claim-to-evidence mapping for every final claim (`src/voice_agent/mcp_servers/gemini_grounding.py:207`, `pipeline/src/voice_pipeline/doc_answer.py:109`, `pipeline/src/voice_pipeline/agent_server.py:741`). |
| **E2** | **Partial** | The specific no-search hedge abstains, but Tavily’s nearest-result hedge still speaks a result and final model text is not universally gated (`src/voice_agent/mcp_servers/gemini_grounding.py:326`, `src/voice_agent/mcp_servers/tavily_search.py:283`, `pipeline/src/voice_pipeline/agent_server.py:741`). |
| **E3** | **Partial** | A web result can lead with a source, but a missing source can be appended after the answer; a source name may be unavailable (`src/voice_agent/mcp_servers/gemini_grounding.py:238`, `pipeline/src/voice_pipeline/agent_server.py:851`). |
| **E4** | **Missing** | Document date and undated status are not retained or spoken by the document-answer path (`src/doc_chat/corpus.py:38`, `pipeline/src/voice_pipeline/doc_answer.py:75`). |
| **E5** | **Partial** | A background difference can produce a second-answer notice, but the initial answer is already emitted and the notice does not guarantee both sources and dates (`pipeline/src/voice_pipeline/agent_server.py:612`, `src/voice_agent/mcp_servers/tavily_search.py:672`). |
| **E6** | **Partial** | Research results are labelled untrusted; the document-answer request lacks equivalent explicit labelling and enforcement (`src/voice_agent/mcp_servers/tavily_search.py:196`, `pipeline/src/voice_pipeline/doc_answer.py:95`). |
| **E7** | **Partial** | Empty cancelled lookups are retried, but failure handling is string-based and a nonempty unrecognized failure can be reused (`pipeline/src/voice_pipeline/agent_tools.py:185`, `pipeline/src/voice_pipeline/agent_tools.py:1054`). |
| **E8** | **Partial** | The web tool retries a classified changing question that did not search, but the front agent can answer without invoking the web tool; the final audit is after emission (`src/voice_agent/mcp_servers/tavily_search.py:572`, `pipeline/src/voice_pipeline/agent_server.py:875`). |

### Search requirements

| ID | Status | Evidence and gap |
|---|---|---|
| **S1** | **Partial** | Enforced within part of `web_answer`, not across every front-agent answer path (`src/voice_agent/mcp_servers/tavily_search.py:558`, `pipeline/src/voice_pipeline/agent_server.py:741`). |
| **S2** | **Partial** | The empty cancellation race is handled; the broader failure contract remains dependent on text matching and nonempty callback results (`pipeline/src/voice_pipeline/agent_tools.py:1060`, `pipeline/src/voice_pipeline/agent_tools.py:1085`). |
| **S3** | **Partial / unverified** | The draft reports a 4 ms gate and a roughly one-second retry goal, but this review did not measure end-to-end latency or establish the bound across paths (`docs/SEARCH-SPEC.md:28`, `docs/SEARCH-SPEC.md:44`, `src/voice_agent/mcp_servers/tavily_search.py:69`). |
| **S4** | **Done for the specified zero-search hedge** | The literal hedge contains no guess. The separate Tavily closest-result behavior still conflicts with the broader abstention principle (`src/voice_agent/mcp_servers/gemini_grounding.py:326`, `src/voice_agent/mcp_servers/tavily_search.py:283`). |
| **S5** | **Partial** | The document prompt requests `NONE`; a failed document race falls back to web rather than necessarily saying the documents are silent and offering a lookup. Direct Case C asks the model to say it does not know (`pipeline/src/voice_pipeline/doc_answer.py:19`, `pipeline/src/voice_pipeline/agent_server.py:389`, `pipeline/config/eval/system_agent.txt:20`). |
| **S6** | **Partial** | Clarification is a prompt rule; there is no general measured false-premise or ambiguity gate (`pipeline/config/eval/system_agent.txt:24`, `pipeline/config/eval/system_agent.txt:26`). |
| **S7** | **Partial** | Source-first wording exists for some grounded answers; server fallback is trailing and unavailable if no source is parsed (`src/voice_agent/mcp_servers/gemini_grounding.py:238`, `pipeline/src/voice_pipeline/agent_tools.py:382`). |
| **S8** | **Partial** | Same-sentence attribution for named-person actions is requested in the prompt; `reportedly` is an example rather than a deterministic check, and an earlier log contains a counterexample (`pipeline/config/eval/system_agent.txt:34`, `pipeline/out/agent_server/20261116-104540.jsonl:8`). |
| **S9** | **Missing** | The document race discards source metadata and has no date metadata (`pipeline/src/voice_pipeline/doc_answer.py:75`, `src/doc_chat/corpus.py:47`). |
| **S10** | **Missing** | The server filters presentation and logs calls, but does not compare final spoken names, dates, and numbers with question and evidence (`pipeline/src/voice_pipeline/agent_server.py:741`, `pipeline/src/voice_pipeline/agent_server.py:870`). |
| **S11** | **Missing** | There is no final-answer numeric support comparator. The document checker’s number matching is a different, substring-based heuristic (`pipeline/src/voice_pipeline/doc_answer.py:60`). |
| **S12** | **Partial** | A recognizable lookup claim can trigger `web_verify`, but raw research results need not have the source form that `turn_claim` recognizes, and the hook is not specifically guaranteed for named-person research answers (`pipeline/src/voice_pipeline/agent_tools.py:370`, `pipeline/src/voice_pipeline/agent_tools.py:406`, `pipeline/src/voice_pipeline/agent_server.py:868`). |
| **S13** | **Partial** | `DIFFERS` can yield a later correction, but there is no guarantee that the notice includes both answers, both source identities, and dates (`src/voice_agent/mcp_servers/tavily_search.py:672`, `pipeline/src/voice_pipeline/agent_server.py:631`). |

### Document requirements

| ID | Status | Evidence and gap |
|---|---|---|
| **D1** | **Missing** | Basename is the only corpus metadata shown; file and chunk identity, dates, hashes, positions, and ingestion time are absent (`src/doc_chat/corpus.py:38`, `src/voice_agent/mcp_servers/docs.py:40`). |
| **D2** | **Missing** | The index is built at server construction; there is no query-time change detection or stale-index response in this path (`src/voice_agent/mcp_servers/docs.py:27`). |
| **D3** | **Missing** | Individual unreadable and empty files are skipped; unsupported extensions are not discovered (`src/doc_chat/corpus.py:30`, `src/doc_chat/corpus.py:38`). |
| **D4** | **Partial** | A special path requires a normalized matching quote, but passages lose source identity, have no IDs or dates, and direct Case C does not use that check (`pipeline/src/voice_pipeline/doc_answer.py:53`, `pipeline/src/voice_pipeline/doc_answer.py:75`, `pipeline/config/eval/system_agent.txt:20`). |
| **D5** | **Missing** | The race chooses a document answer and cancels the web task; it does not compare both answers for changing questions (`pipeline/src/voice_pipeline/agent_server.py:382`). |
| **D6** | **Missing** | There is no conflict reconciliation with both file names and dates in the document-answer result (`pipeline/src/voice_pipeline/doc_answer.py:109`). |
| **D7** | **Missing** | Document passages lack an explicit untrusted-data wrapper in the document-answer request; this review did not find an end-to-end adversarial-file acceptance test (`pipeline/src/voice_pipeline/doc_answer.py:95`). |
| **D8** | **Missing** | No evidence-date versus question-time comparison exists in the shown document flow (`src/voice_agent/mcp_servers/docs.py:40`, `pipeline/src/voice_pipeline/doc_answer.py:95`). |

## 5. Evaluation and observability

### What exists

- `search_eval.py` has eight search cases and checks required words or alternatives in tool text. It records duration, but keyword presence cannot establish entailment, correct attribution, or absence of a false premise (`config/eval/search_cases.jsonl:1`, `scripts/search_eval.py:1`, `scripts/search_eval.py:46`).
- `chain_eval.py` has scenario-based trajectory checks, including lookup count, state, delegation, source and follow-up checks; it has a fake offline runner and a separate live runner (`scripts/chain_eval.py:102`, `scripts/chain_eval.py:252`, `scripts/chain_eval.py:632`, `scripts/chain_eval.py:776`).
- The pipeline dry run scores expected calls and intents, leakage, and timing. Its call matcher collapses repeated calls to the same tool, so it is not sufficient by itself to detect repeated searches (`pipeline/src/voice_pipeline/dryrun.py:144`, `pipeline/src/voice_pipeline/dryrun.py:166`, `pipeline/src/voice_pipeline/dryrun.py:294`).
- `freshqa_gate_check.py` measures classifier flags against supplied FreshQA labels; it explicitly warns that the dataset is optimistic for real traffic. It is a gate evaluation, not a final-answer or false-premise evaluation (`scripts/freshqa_gate_check.py:1`, `scripts/freshqa_gate_check.py:90`).
- `output_audit.py` checks repeated status sentences, premature job claims, duplicate jobs, and short fragment turns; it does not check factual support (`scripts/output_audit.py:1`, `scripts/output_audit.py:42`).
- Unit tests cover the document checker, changeability logging, routing, hedging, filters, and other components. The specific document tests show that uppercase and formatting-normalized quotes are accepted, and source identities are intentionally discarded by `passages_of` (`pipeline/tests/test_doc_answer.py:23`, `pipeline/tests/test_doc_answer.py:81`).
- Server logs contain question, calls, answer, and timing; grounding and verification ledgers add search counts and verdicts. The server log does not by itself record a complete evidence tree or final claim-to-source relationships (`pipeline/src/voice_pipeline/agent_server.py:870`, `src/voice_agent/mcp_servers/gemini_grounding.py:391`, `src/voice_agent/mcp_servers/tavily_search.py:657`).

The grounding ledger contained **226 zero-search rows among 414 non-failure rows** in this read. That is a mixture of historical questions, configurations, and code states; settled-fact calls may legitimately have zero searches under the current tool prompt. It does **not** measure the current rate of unsafe changing answers (`~/.local/state/voice_agent/web_answer_ledger.jsonl:1`, `~/.local/state/voice_agent/web_answer_ledger.jsonl:418`, `src/voice_agent/mcp_servers/gemini_grounding.py:47`).

### What to build

1. **A labelled benchmark tied to requirements.** Start with the 12 document cases in the spec, then add real paraphrases, multi-turn corrections, adversarial instructions, missing/failed sources, and known-good home/chat cases. Each case should contain the corpus snapshot, question, expected claims or abstention, permitted evidence IDs, time scope, and requirement IDs (`docs/SEARCH-SPEC.md:74`; `evaluation/source-notes/M2_observing-and-benchmarking-agents.md §Core concepts`; `agentic-ai/source-notes/M4_evals-error-analysis.md §Core concepts`).

2. **Separate component and end-to-end metrics.** Measure ingestion coverage/errors; retrieval recall@k and precision@k against labelled passages; answer faithfulness, relevance, citation correctness, abstention recall, false-premise handling, and stale-wins rate; then measure the text actually delivered to the client. A correct tool call and a correct final answer are separate outcomes (`RAG/source-notes/M2_information-retrieval-foundations.md §7`; `evaluation/source-notes/M4_actions-conversations-and-outcomes.md §Core concepts`; `ragbook.txt ch. 6, “RAG Evaluation Metrics”`).

3. **Trace the decision chain.** Add `session_id`, `turn_id`, parent/span IDs, configuration and prompt version, route and gate decisions, source and chunk IDs, retrieval and search status, cache key/hit, claim-to-evidence links, verification verdict, emitted text, latency, and usage. Record the distinction between a skipped, failed, cancelled, and successful step. Define redaction and retention before storing passage content. The reading’s span/trace/session distinction is directly useful here (`evaluation/source-notes/M1_why-evals-matter-and-traces.md §Core concepts`; `ragbook.txt ch. 7, “Evaluation and Observability with AI Agents”`).

4. **Calibrate semantic judges rather than treating them as ground truth.** Deterministic checks should cover IDs, order, dates, counts, and required fields. Where entailment or answer relevance needs a judge, use separate single-purpose rubrics and a human-labelled held-out set; report false negatives and false positives, not only aggregate agreement (`evaluation/source-notes/M3_building-evaluators-you-can-trust.md §Core concepts`).

5. **Compare one change at a time.** Run candidate and baseline on the same cases, including known-good cases, and report quality, p50/p90 latency, search calls, tokens, and cost separately. The notebook extraction `L6.txt §“First we run … baseline”` illustrates a baseline-and-compare workflow; `evaluation/source-notes/M5_improving-agentic-applications.md §Core concepts` recommends controlled comparisons.

## 6. Recommendations, in order

The thresholds below are proposed acceptance criteria for a seeded test set, not claims of current performance.

### 1. Put a factual-answer gate before emission

**Change:** Introduce a shared typed result for factual turns: question, temporal scope, status, claims, evidence references, and source dates. Route every factual path—including fast router results, hedged Gemini, direct `search_docs`, web results, stall fallback, and document race—through a release decision. Allow immediate progress speech such as “Let me check,” but buffer factual claims until supported or abstain. Treat a search call, an empty result, and a supported answer as different states.

**Touches:** `pipeline/src/voice_pipeline/agent_server.py:741`, `pipeline/src/voice_pipeline/agent_tools.py:1018`, `pipeline/src/voice_pipeline/hedge.py:138`, `src/voice_agent/mcp_servers/tavily_search.py:558`.

**Expected effect:** Closes the most important gap between logging an unsupported answer and preventing its release. It also gives later RAG work one contract to implement.

**Trade-off:** Buffering can delay the first factual sentence; a deterministic validation pass is cheap, while semantic verification or an additional lookup increases latency and cost. Keep progress speech separate from answer latency.

**Acceptance test:** In offline fake-provider tests, inject no lookup, zero-search, failed, cancelled, empty, and missing-source results through each route. **100%** must emit either a supported answer with its evidence record or the designated abstention; **0** may emit the injected unsupported fact. Include the current `unchecked` test’s scenario as a release test, not merely a logging test.

**Reading:** `evaluation/source-notes/M4_actions-conversations-and-outcomes.md §Core concepts` separates process from outcome; `agentic-ai/source-notes/M4_evals-error-analysis.md §Core concepts` supports end-to-end and component tests.

### 2. Preserve evidence and check answer meaning

**Change:** Keep document ID, chunk ID, offsets, exact source text, document date status, and the quoted span through retrieval, generation, and output. Require each factual claim to reference evidence. Check exact quote offsets deterministically; evaluate whether each claim follows from its cited span with a calibrated semantic check or a constrained extractive answer. Handle negation, multiple claims, and silent passages explicitly. Apply the same contract to the direct Case C path.

**Touches:** `src/voice_agent/mcp_servers/docs.py:40`, `pipeline/src/voice_pipeline/doc_answer.py:53`, `pipeline/src/voice_pipeline/doc_answer.py:75`, `pipeline/config/eval/system_agent.txt:20`.

**Expected effect:** Makes citations inspectable and reduces answers that contain a real quote but an unsupported or contradictory interpretation.

**Trade-off:** More structured output and a possible second model or human-calibrated judge add latency and complexity. Exact-quote checking is cheap; semantic verification is the expensive part.

**Acceptance test:** On a labelled fixture of at least 50 claim/evidence pairs, including negation, swapped entities, unrelated real quotes, first-word names, numeric formatting, and two-source answers: **100%** of cited spans resolve exactly; semantic-check **recall for unsupported claims ≥95%**, with precision reported separately. On all silent-passage fixtures, no factual answer is released.

**Reading:** `ragbook.txt ch. 6, “Generation Failures”`; `RAG/NOTES.md §Settled Terminology & Taxonomy` distinguishes faithfulness from answer relevance.

### 3. Repair lookup identity, failure handling, and attribution

**Change:** Replace string-matched success/failure with structured statuses. Key in-flight and cached results by normalized query **plus temporal scope and relevant entities**; do not reuse a different query merely because it occurs in the same turn. Remove the closest-result answer when required query attributes are unconfirmed. Preserve source IDs and dates with each claim, and place attribution with the claim rather than relying on a trailing generic source sentence.

**Touches:** `pipeline/src/voice_pipeline/agent_tools.py:185`, `pipeline/src/voice_pipeline/agent_tools.py:1054`, `src/voice_agent/mcp_servers/tavily_search.py:87`, `src/voice_agent/mcp_servers/tavily_search.py:250`, `src/voice_agent/mcp_servers/tavily_search.py:283`, `src/voice_agent/mcp_servers/gemini_grounding.py:111`.

**Expected effect:** Reduces wrong-question reuse, accidental “already looked up” success, and misleading source coverage.

**Trade-off:** Fewer approximate cache hits and more searches for genuinely different questions; likely higher latency and search spend on corrections. Exact duplicates can still share work.

**Acceptance test:** For a fixture matrix covering present/past, different entities, corrected questions, distinct research and answer queries, failures, and cancellation: **0 cross-query cache hits**, **0 failed results reported as successful**, and **100%** of released factual claims have a resolvable cited source. Repeated identical requests should still coalesce into one provider call.

**Reading:** `context-engineering/source-notes/M1_intro-context-engineering.md §4` explains poisoning and confusion; `evaluation/source-notes/M4_actions-conversations-and-outcomes.md §Core concepts` covers argument and value propagation.

### 4. Build a small, auditable ingestion lifecycle

**Change:** Define supported formats and an explicit ingest operation. Record canonical path or stable file ID, content hash, modification time, ingestion time, extraction status, content dates with provenance, version, and chunk IDs/positions. Report each failed, empty, or unsupported file. On query, compare the indexed version with the source and re-index or say the index is stale. Treat modification time as metadata, not as the date of every claim.

**Touches:** `src/doc_chat/corpus.py:14`, `src/voice_agent/mcp_servers/docs.py:27`.

**Expected effect:** Makes ingestion failures and changed files visible and makes citations reproducible.

**Trade-off:** Hashing and re-indexing cost CPU and storage; checking every large file on every query may be too expensive. A manifest plus incremental checks can reduce repeated work, but the chosen freshness guarantee must be explicit.

**Acceptance test:** A temporary seeded corpus containing duplicate basenames, a changed file, an empty file, an unreadable file, and an unsupported format must produce a named status for **100%** of files. After a change, **0** answers may silently use the old indexed content. Every returned chunk must resolve to one unique file version and position.

**Reading:** `ragbook.txt ch. 3, “Advanced Data Ingestion”` and ch. 6, “Failures Due to Inadequate Data Ingestion”; `context-engineering/source-notes/M2_retrieving-context.md §Core concepts` on metadata and retrieval diagnosis.

### 5. Make time and disagreement explicit

**Change:** Resolve “now,” “latest,” and historical scopes against the injected date. For changing document claims, obtain a current grounded look when available and retain both results. Report agreement or disagreement with source names and dates; if freshness or source dates cannot be established, say so rather than selecting a winner. Decide how to handle unavailable or exhausted verification before presenting a current claim.

**Touches:** `pipeline/src/voice_pipeline/agent_server.py:364`, `pipeline/src/voice_pipeline/agent_server.py:612`, `src/voice_agent/mcp_servers/tavily_search.py:632`, the future ingestion metadata.

**Expected effect:** Prevents a faithful quotation of stale material from being presented as current, and makes disagreements reviewable.

**Trade-off:** A changing document answer may require an extra search and wait. Running the checks concurrently helps latency but increases provider usage.

**Acceptance test:** In a seeded matrix of current, stale, undated, conflicting, and unavailable-web cases: **0 stale claims presented as current**; **100%** of conflict cases identify both sources and their known dates or explicitly say a date is unknown. Measure document-turn p50 and p90 against the spec’s proposed latency target (`docs/SEARCH-SPEC.md:93`).

**Reading:** `ragbook.txt ch. 6, “Generation Failures”`; `context-engineering/source-notes/M1_intro-context-engineering.md §4`.

### 6. Add adversarial-data and correction-memory tests

**Change:** Label document and web content as untrusted in every model request, keep instructions outside those blocks, and restrict what retrieved text can cause tools to do. Preserve correction notices as structured conversational events so subsequent turns do not silently rely on a superseded answer. Continue to label chain facts as unverified until they have evidence references.

**Touches:** `pipeline/src/voice_pipeline/doc_answer.py:95`, `pipeline/src/voice_pipeline/scratchpad.py:18`, `pipeline/src/voice_pipeline/agent_tools.py:1140`, `src/text_chat/cli.py:64`.

**Expected effect:** Reduces instruction/data confusion and prevents known corrections from disappearing from the next model context.

**Trade-off:** Small prompt and state overhead; adversarial evaluation requires a maintained fixture set. A label alone does not guarantee prompt-injection resistance.

**Acceptance test:** In at least 30 seeded instruction-in-document and instruction-in-web cases, **0** injected instructions may alter tool permissions or the requested answer. In a multi-turn correction fixture, **100%** of next-turn requests must carry the correction or an explicit supersession record.

**Reading:** `context-engineering/source-notes/M1_intro-context-engineering.md §4`; `ragbook.txt ch. 7, “Evaluation and Observability with AI Agents”`.

### 7. Establish the benchmark and traces before tuning retrieval

**Change:** Implement the evaluation and trace plan above, then use failures to choose chunking, top-k, query rewriting, or reranking. Capture actual emitted text and separate retrieved-evidence quality from answer quality.

**Touches:** `scripts/search_eval.py:41`, `scripts/chain_eval.py:571`, `pipeline/src/voice_pipeline/dryrun.py:166`, `pipeline/src/voice_pipeline/agent_server.py:870`.

**Expected effect:** Makes improvements measurable and prevents a better-looking answer from hiding a worse retrieval or abstention rate.

**Trade-off:** Initial annotation and instrumentation time; offline deterministic tests are cheap, while semantic and live evaluations incur model and search costs.

**Acceptance test:** Every benchmark case has a requirement ID and expected behavior; every factual turn has a trace with linked route, evidence status, emitted answer, and latency. On the seeded corpus, use the spec’s proposed **≥95% abstention recall**, **0 stale-wins**, and **100% source coverage** as initial targets, reported with case counts (`docs/SEARCH-SPEC.md:93`).

**Reading:** `agentic-ai/source-notes/M4_evals-error-analysis.md §Core concepts`; `evaluation/source-notes/M2_observing-and-benchmarking-agents.md §Core concepts`; `RAG/source-notes/M2_information-retrieval-foundations.md §7`.

### What I would not do yet

- **Do not add a vector database solely because the next feature is called RAG.** The spec explicitly excludes one, and the current BM25 retriever provides a baseline. Add dense or hybrid retrieval only if labelled recall and precision expose a problem it addresses (`docs/SEARCH-SPEC.md:11`, `src/voice_agent/mcp_servers/docs.py:27`; `RAG/source-notes/M3_information-retrieval-in-production.md §Core concepts`).
- **Do not treat a quote, matching capitalized tokens, a search count, or two agreeing model answers as proof of truth.** Each is a useful signal with a narrower guarantee than entailment and freshness (`pipeline/src/voice_pipeline/doc_answer.py:53`, `src/voice_agent/mcp_servers/tavily_search.py:645`; `ragbook.txt ch. 6, “Generation Failures”`).
- **Do not add persistent episodic memory before measuring the need.** The current four-pair history and task state already have identifiable provenance and correction gaps. First test returning threads and corrections; otherwise a memory system can retain unsupported claims (`src/text_chat/cli.py:34`, `pipeline/src/voice_pipeline/chain_state.py:140`; `context-engineering/source-notes/M4_storing-context.md §Core concepts`).
- **Do not optimize the benchmark to keyword presence alone or fine-tune first.** Keyword checks can pass a wrong answer, and the readings recommend inspecting failures and improving simpler components before expensive model changes (`scripts/search_eval.py:46`; `agentic-ai/source-notes/M4_evals-error-analysis.md §Core concepts`; `evaluation/source-notes/M5_improving-agentic-applications.md §Core concepts`).

## 7. Open questions for the owner

1. Which formats must the first ingest support, where do files arrive, and is ingestion explicit or watched? The spec leaves this open (`docs/SEARCH-SPEC.md:124`).
2. For a question that asks only what a document says, should the answer always remain document-scoped, with a separate offer to check current reality? For a question asking what is current, is a web check mandatory even if the user did not request one?
3. When a second look differs, should the correction interrupt the user, appear before the next prompt, or block the initial factual answer? The current text behavior displays notices before a later prompt (`src/text_chat/cli.py:64`; `docs/SEARCH-SPEC.md:126`).
4. What is the acceptable p50/p90 latency and search/model budget for evidence-gated answers? The spec’s current latency figures are proposed targets (`docs/SEARCH-SPEC.md:93`).
5. What evidence dates count as sufficient for each class of claim: source publication date, explicit “as of” date, retrieval date, document modification time, or a combination? What should happen when all are absent?
6. How should the system handle contradictory authoritative sources and unavailable verification: report both, abstain entirely, or give a dated document-scoped answer?
7. What privacy and retention policy should govern traces, retrieved passages, uploaded files, and correction history? The current logs include questions, tool arguments, and answers (`pipeline/src/voice_pipeline/agent_server.py:870`).