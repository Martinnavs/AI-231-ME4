# Implementation Plan: context-p1-p3 (CONTEXT-CRITIQUE P1-P3)

## Requirements
- **P1:** a grounded `web_answer` must show that a search happened and must give a source.
  - Put OnIt's rule in the Gemini grounding prompt: "anything that can change needs a tool result, not memory; who holds a job is an example".
  - Keep the source titles and the support segments from the response.
  - Log the query in the ledger.
  - When a changeable question comes back with 0 searches: retry once with a forced-search instruction, then fall back to Tavily, then speak a hedge.
  - Never cache an unsearched answer to a changeable question.
  - Say the source in words.
  - The check is code on `groundingMetadata`, with no extra LLM call.
  - Tests use saved real Gemini responses.
- **P2:**
  - D2: one turn adds 1 to the chain's unrelated count, not 2.
  - D3: the web_research -> web_answer redirect keeps the model's resolved query.
  - D4: the "No, I mean ..." and "Got it / thanks" rules move into the base prompt.
  - D9: "turn it up/down" right after a temperature or brightness change no longer always means volume.
- **P3:**
  - A short date/time/place line on every Gemini user turn, hedged turns included. The city is the fallback when the barangay lookup failed.
  - An explicit "you cannot" block in the front prompt.
  - The venue goes into the grounding prompt.
- **Owner constraints:**
  - No over-engineering.
  - No live Gemini/Tavily call without stating its cost first.
  - Never read or print secrets.
  - No push. Commit only by explicit file name.
  - The ruff, mypy and pytest gate must pass.
  - Volatile content goes last.
  - The per-turn line is short and must not break prompt-prefix caching.

## Repo / Feature / Files / Acceptance Criteria / Non-Goals
- **Repo/area:** `/mnt/School/PhD/AI231/AI-231-ME4`, branch `context-engineering`.
  - git needs `git -c safe.directory=/mnt/School/PhD/AI231/AI-231-ME4 ...`.
  - Root package `src/voice_agent`: MCP servers, root `.venv`, gate `make check test`.
  - `pipeline/` package `voice_pipeline`: own `.venv`, gate `make pipeline-check pipeline-test`.
- **Feature:** context-p1-p3.
- **Files:**
  - Root:
    - `src/voice_agent/mcp_servers/gemini_grounding.py`
    - `src/voice_agent/mcp_servers/tavily_search.py`
    - `scripts/capture_grounding.py` (new)
    - `tests/fixtures/grounding/*.json` (new)
    - `tests/mcp_servers/test_gemini_grounding.py`
    - `tests/mcp_servers/test_tavily_search.py`
  - Pipeline:
    - `pipeline/src/voice_pipeline/agent_tools.py`
    - `pipeline/src/voice_pipeline/agent_server.py`
    - `pipeline/src/voice_pipeline/venue.py`
    - `pipeline/src/voice_pipeline/fast_path.py`
    - `pipeline/config/eval/system_agent.txt`
    - `pipeline/config/eval/system_agent_case_f.txt`
    - `pipeline/config/eval/agent_prompts.jsonl`
    - `pipeline/tests/test_chain_wiring.py`
    - `pipeline/tests/test_prompt_scoping.py`
    - `pipeline/tests/test_venue.py`
    - `pipeline/tests/test_agent_server.py` (or a new `pipeline/tests/test_turn_context.py`)
- **Acceptance criteria:** listed per task below. All of them are checked by automated tests, except the owner-priced live checks, which are marked optional.
- **Non-goals:** P4-P8 (history annotation, usage metrics, threads, episodes, notes tools).
  - No sources in the turn log or NOTES.md (P4).
  - No LLM verifier.
  - No "support segment covers the answer" gate. The gate is `searches == 0`, as the owner specified.
  - No router (350M) prompt changes. No Now line for the router or the fast path.
  - No `dryrun.py` harness changes.
  - No `--history` changes.
  - No Wikipedia-path source phrase.
  - No barangay in the grounder prompt (city only).
  - No new dependencies.

## Developer Tasks

### Task 0: grounding request builder, new prompt text, fixture capture script (Files: `src/voice_agent/mcp_servers/gemini_grounding.py`, `scripts/capture_grounding.py`)
- **Action:**
  1. In `gemini_grounding`, change `SYSTEM`:
     - Replace "Do not search for settled facts" with OnIt's rule, adapted: "Anything that can change needs a search result, not memory: who holds a job or office, prices and rates, versions, rankings, records, results, anything latest/last/next/current/new."
     - Settled facts (capitals, authors, historical dates) still need no search.
     - Add a `{place}` slot: "The user is in {place}; 'here' means there."
  2. Add a `FORCE_SEARCH` sentence ("Search the web before you answer this question; do not answer from memory.") that is appended to the system text on a retry.
  3. Split the body construction out of `ask()` into a pure function, so the request is built in one place and the capture script reuses it. The function takes `query`, `today`, `place`, `research` and `force`.
  4. `ask()` gains `place: str = DEFAULT_PLACE` (`"Cebu City"`) and `force: bool = False`.
  5. New `scripts/capture_grounding.py`, about 50 lines:
     - Arguments: a list of `id=question` cases, an output dir (default `tests/fixtures/grounding/`), `--legacy-prompt` (sends the pre-change `SYSTEM` text, kept as a literal in the script, so a real zero-search answer to a changeable question can be captured), `--force` (the retry body), and `--yes`.
     - Without `--yes` it prints the case count and the estimated cost, and exits 0 with no network call.
     - It reads the key only through `gemini_grounding.load_key()` and sends it only as the header.
     - It writes `{"id", "query", "system", "force", "captured_at", "response": <verbatim response.json()>}`.
     - Before writing, it asserts that the key string is not in the serialized text and refuses to write if it is.
     - It never prints the key, headers or the request.
     - It appends a ledger row through `Ledger` so the month's search count stays true.
- **Established (verified):**
  - `SYSTEM` holds "Do not search for settled facts", and `ask()` sends `thinkingLevel` `minimal` (read in `gemini_grounding.py`).
  - The key is read from `$GEMINI_API_KEY` or `~/.secrets/google-ai-studio/api-key.txt` and sent as `x-goog-api-key` (`load_key`, `ask`).
  - The response JSON carries no key.
  - `parse()` keeps only titles. `Grounded` has no URIs or supports.
  - OnIt's wording is at `onit/mcp/prompts/prompts.py`, in the "Recency and source quality" block (~line 364).
- **Must verify:**
  - `uv run pytest tests/mcp_servers -q` creates the root `.venv` (absent in ME4). If uv needs the network to resolve, report it; it is free, but it is not offline.
  - The no-`--yes` run makes zero HTTP requests (unit test with a transport that fails on any request).
- **Unknowns:** none beyond the capture itself (see the human gate).
- **Dependencies/license:** none new. Uses httpx, already a dependency.
- **Symbols:** `gemini_grounding.SYSTEM`, `SYSTEM_RESEARCH`, `ask`, `load_key`, `Ledger`, `GEMINI_URL`.
- **Acceptance criteria:**
  - `SYSTEM` contains the OnIt-style rule and `{place}`, and no longer contains "Do not search for settled facts".
  - `ask(force=True)` sends `FORCE_SEARCH` in the system text. `ask(place="X")` sends "X". Both are tested with the existing `Both`-style transport.
  - The script refuses to call the network without `--yes`, and the key-in-output guard is unit-tested with a fake key.
  - The root ruff, mypy and pytest gate is green.
- **Non-goals:** the gate, sources, ledger columns (Task 1), the capture run itself (human gate).
- **Security-sensitive:** yes. The script handles the API key and writes files to the repo; they must never contain the key.

**HUMAN GATE between T0 and T1:** before any live call, the orchestrator relays the capture cost to the owner (see Decisions). After approval, the capture runs once. Recommended case set, about 12 calls:
- New prompt (`SYSTEM`): `nvidia_ceo` "Who's the CEO of Nvidia?", `up_president` "Who is the president of the University of the Philippines?", `nba_last` "Who won the last NBA championship?", `cebu_mayor` "Who is the mayor here?" (place = Cebu City), `python_latest` "What is the latest stable version of Python?", plus settled facts `france_capital`, `noli_author` ("Who wrote Noli Me Tangere?") and `mona_lisa`.
- `--legacy-prompt`: `nvidia_ceo_legacy` and `up_president_legacy`. The ledger says 60% of these come back with 0 searches. Re-run once if a legacy case happens to search; that adds at most 2 calls.
- `--force`: `nvidia_ceo_forced` and `up_president_forced`.

If no real zero-search changeable response is obtained in 2 tries, report it. Do not hand-build one.

### Task 1: the search gate, sources, ledger query, venue flag (Files: `src/voice_agent/mcp_servers/gemini_grounding.py`, `src/voice_agent/mcp_servers/tavily_search.py`, `tests/mcp_servers/test_gemini_grounding.py`, `tests/mcp_servers/test_tavily_search.py`, `tests/fixtures/grounding/*.json` (read-only, from the capture))
- **Sequencing:** after Task 0 and the capture. Resume the same developer instance.
- **Action:**
  1. `gemini_grounding`:
     - `Grounded` gains `sources: list[tuple[str, str]]` (title, uri) and `supports: list[list[int]]` (each support's `groundingChunkIndices`).
     - `parse()` returns them.
     - New pure function `changeable(question) -> bool`: one case-insensitive regex. Role nouns: ceo, president, chair(man|woman|person), head, mayor, governor, coach, minister, senator, secretary, director, dean, chancellor, leader, captain, owner, champion(s), winner. Time words: current(ly), now, latest, last, next, new(est), recent, today, this year|week|month. Changeable things: price, rate, version, release, ranking, record, score, result.
     - New `publisher(title) -> str | None`, spoken words for a chunk title or URL host:
       - A bare domain loses `www.` and its generic/country suffix labels (com, org, net, gov, edu, ph, io, co, uk, ...), keeps the last remaining label and title-cases it. Labels of 3 letters or fewer become upper case ("up.edu.ph" -> "UP"), and "wikipedia" -> "Wikipedia".
       - A page title "X - Site" or "X | Site" gives the last segment if it is 4 words or fewer.
       - Otherwise None. The text goes through `tavily_search.clean_text`-style sanitising (or an equivalent local one) and is capped at 40 chars.
     - `Ledger.add(grounded, query, forced=False)` writes `query` (first 200 chars), `forced` (only when true) and `sources` (at most 3 titles, 60 chars each). `_read` must still accept old rows.
     - `main`-less: `DEFAULT_PLACE = "Cebu City"`.
  2. `tavily_search.build_server(..., place=DEFAULT_PLACE)`. `main()` parses `--home` (default "Cebu City") and passes it. `grounded()` passes `place` to `ask`.
  3. The gate in `grounded()`/`search()`:
     - (a) The first answer has `searches >= 1`, or `not changeable(query)`: the path is unchanged (cached as now) plus a source sentence when `sources` exist.
     - (b) `changeable(query)` and `searches == 0`: one retry `ask(force=True)`, bounded by the remaining time of a total grounded budget (decision: `GROUNDED_TOTAL_S = 4.5` from the first call). A retry with `searches >= 1` is cached and returned with its source.
     - (c) Otherwise fall through to Tavily, keeping the unsearched text as a guess. A Tavily answer gets the existing `unconfirmed()` hedge and a source sentence from the first result's URL host, and is cached as now.
     - (d) Tavily fails (no key, reserve mode, HTTP error, timeout, no answer) and a guess exists: return `HEDGE.format(guess)`, NOT cached. With no guess, the existing Wikipedia fallback.
     - A timeout or HTTP error on the first Gemini call does not retry. It goes to Tavily as today.
  4. The source sentence is appended after the 2-sentence/400-char cut: `" Source: {name}."` (decision). The chunk is chosen from the first support's first chunk index, else chunk 0. With no derivable name, no sentence on the cached grounded path.
  5. `HEDGE` wording, default: "I couldn't confirm that with a search; my best guess is: {guess}". It must NOT match `voice_pipeline.agent_tools.FAILURE` (`couldn't find|did not answer|not connected|can't say|error`).
- **Established (verified):**
  - The ledger has 350 rows. 348 are not failed, and 208 of those have 0 searches. The median time is 0.91 s with 0 searches against 1.80 s with 1 (p90 2.48 s). 151 searches were used in 2026-11. Tokens: median 138, max 277. The row keys are only month, searches, seconds, time and tokens (computed from `~/.local/state/voice_agent/web_answer_ledger.jsonl`).
  - `grounded()` caches every reply through `remember()` with no check. `grounded_research()` keeps titles. `GEMINI_TIMEOUT_S` is 3.0, and Tavily's `TOTAL_TIMEOUT_S` is 3.5.
  - The MCP child cannot import `voice_pipeline`: they are separate packages and venvs. So `changeable()` belongs in `voice_agent`, not next to `agent_tools.one_fact_question`. The critique's "Where" is wrong on this point.
  - `web_answer` is in `SPOKEN_RESULT_TOOLS`, so its text is spoken verbatim.
  - `SourceFilter`/`strip_sources` drop only parenthesised "(source..." asides (`payload_guard.py`), so a plain "Source: X." sentence survives.
  - A router-path result that matches `FAILURE` makes `run_routed` return None and escalate to Gemini (`agent_server.run_routed`). That would mean a second full lookup.
  - The existing test infrastructure (the `Both` transport and fastmcp `Client` in `tests/mcp_servers/test_gemini_grounding.py`) can replay fixtures.
- **Must verify:**
  - Against the real fixtures:
    - chunk title shapes (bare domain or page title) -> `publisher()` output for every captured title;
    - whether `groundingSupports` exist in flash-lite responses (if absent, chunk 0);
    - whether the `--force` responses show `searches >= 1`. If not, report it; the retry is then useless and the plan's fallback order needs revisiting.
  - Hedge vs `FAILURE`: a root test holds a literal copy of the `FAILURE` pattern, with a comment naming `agent_tools.FAILURE`, and asserts `HEDGE` and the source sentence do not match it.
  - Cache: a changeable 0-search answer that ends in a hedge is not returned from the cache on the next identical call. That second call hits the network again (count the transport requests).
- **Unknowns:**
  - There is no API knob to force `google_search`, only the instruction. Its reliability is known only from the forced fixtures.
  - How many of the 208 zero-search calls were changeable is unknowable until the query column exists.
- **Dependencies/license:** none new.
- **Symbols:** `Grounded`, `parse`, `ask`, `Ledger.add/_write/_read`; `tavily_search.build_server`, `web_answer`, `grounded`, `search`, `remember`, `spoken`, `unconfirmed`, `hedged`, `clean_text`, `main`, `first_sentences`.
- **Acceptance criteria** (all tests against `tests/fixtures/grounding/*.json`, no network):
  - `parse` returns text, searches, sources and supports for every fixture.
  - `changeable` is true for the role, last and latest questions, and false for the 3 settled ones and "Who is Rizal?".
  - Legacy zero-search Nvidia fixture, then forced fixture: 2 Gemini requests, the answer is spoken with "Source: ...", and it is cached.
  - Zero-search, then zero-search (reuse the legacy fixture), then Tavily OK: a Tavily answer with a source, 2 Gemini requests and 1 Tavily request.
  - Zero-search, then zero-search, then Tavily error: `HEDGE` + guess, and a second call is not served from the cache.
  - A settled zero-search fixture: 1 request, no retry, cached, as today.
  - A ledger row has `query` and `forced`, and old rows still sum.
  - The request system text carries the place passed to `build_server`, and `main` parses `--home`.
  - The extra latency is limited to changeable zero-search answers. A test asserts that settled and searched fixtures make exactly 1 Gemini request.
  - The root gate is green.
- **Non-goals:** the turn log, NOTES.md, the Wikipedia source, the research path gate, `scripts/search_eval.py` changes (its live run is an optional owner check).
- **Security-sensitive:** yes.
  - The ledger now persists user questions (local state dir, out of git; cap 200 chars).
  - Untrusted page titles become spoken text: sanitise, cap, no URLs.
  - The key handling is unchanged.

### Task 2: P2 + P3 in the pipeline (Files: `pipeline/src/voice_pipeline/agent_tools.py`, `pipeline/src/voice_pipeline/agent_server.py`, `pipeline/src/voice_pipeline/venue.py`, `pipeline/src/voice_pipeline/fast_path.py`, `pipeline/config/eval/system_agent.txt`, `pipeline/config/eval/system_agent_case_f.txt`, `pipeline/config/eval/agent_prompts.jsonl`, `pipeline/tests/test_chain_wiring.py`, `pipeline/tests/test_prompt_scoping.py`, `pipeline/tests/test_venue.py`, `pipeline/tests/test_agent_server.py` or a new `pipeline/tests/test_turn_context.py`)
- **Sequencing:** runs in parallel with T0/T1. There is no file overlap and no data edge.
- **Action:**
  1. **D2:** delete the `CHAIN.touch_unrelated(...)` call in `agent_tools._recorded`. The end-of-turn count in `AgentServer.handle` becomes the only site; leave it as is.
  2. **D3:** in the `_recorded` redirect, use the model's `params.arguments["query"]` when it is non-empty after stripping. Use `question.current` only when it is empty.
  3. **D4:** move the two Case F bullets ("No, I mean ..." corrects your last action..., and "Got it", "thanks", "mm-hmm"...) verbatim into the base prompt and remove them from `system_agent_case_f.txt`.
  4. **D9:**
     - `fast_path` gets a constant `CONTEXT_DEPENDENT = {"turn it up", "turn it down"}` (normalized forms).
     - In `AgentServer.handle`, after `fast_path.match`: if `found` is a smart_home VOLUME_UP/DOWN from a `CONTEXT_DEPENDENT` phrase, there is an earlier assistant message, and the last entry in `self.record` before this turn is a smart_home call with intent TEMPERATURE or BRIGHTNESS, then drop the fast path and send the turn straight to `self.gemini(self.steer(messages, case_f=False), ...)`, skipping the router.
     - Add one Case A/E sentence to the base prompt: "'Turn it up/down' right after a temperature or brightness change means that setting, not the volume; with no number, ask (Case E)." (decision: ask vs step).
  5. **P3 Now/Place line:**
     - `venue.here() -> str` returns "Barangay X, City" when known, else the city.
     - `AgentServer` gains an opt-in `stamp: Callable[[], str] | None = None`. It is set only by `run()`, which mirrors how `probe`/`test_delay` are set. Existing byte-exact tests stay valid, and new tests set it.
     - The line format, about 25 tokens: `[Now: Saturday, November 14, 2026, 7:05 PM, Asia/Manila. Place: Barangay X, Cebu City.]`. It uses `fast_path.TIMEZONE` and an injectable clock.
     - It is appended last to the last user message:
       - in `gemini()`, which covers the chain, no-router, D9 and escalation paths;
       - in the hedged branch of `routed_turn`, where the hedged requests get ONLY the line (no chain block, no Case F), so they are otherwise byte-identical to today.
     - The router call keeps using the unsteered `messages`.
  6. **P3 prompt:**
     - A static "You cannot" block, at most 6 short lines (decision on the list). Default list: look up personal contact details, buy or pay for anything, remember earlier conversations, see a web page except through your tools, reach the internet when offline.
     - One static sentence: "Each user message ends with a [Now: ...] line: use it for dates and 'here'; never read it out."
     - `{today}` handling for `--builtin-search` is unchanged.
  7. Raise the `test_the_prompt_grew_by_less_than_1900_characters` bound explicitly. The base prompt grows from 6803 to about 7700 chars (about +210 tokens per request, x2 requests when hedged). Add a comment with the new size.
  8. Add dry-run prompts to `agent_prompts.jsonl`:
     - ["Play some jazz.", "No, I mean classical."] expects PLAY_MUSIC jazz, then PLAY_MUSIC classical, with no lookup;
     - ["Who painted the Mona Lisa?", "Got it, thanks."] expects web_answer, then no call;
     - ["Set the temperature to 24 degrees.", "Turn it up."] expects TEMPERATURE 24, then no call (with the ask default);
     - ["Can you order me a pizza?"] expects no call.
- **Established (verified):**
  - D2: both counting sites exist (`_recorded`; the end of `handle`, `touch_unrelated(side=...)`). The repro is at `/tmp/claude-1000/-mnt-School-PhD-AI231-AI-231-ME3/5bf5aa4d-2e40-4732-b93f-2a6ec670c6d7/scratchpad/repro/double_count.py`.
  - `test_unrelated_tools_touch_an_active_chain_but_chain_tools_do_not` (test_chain_wiring) asserts the `_recorded` touch and must be rewritten as `test_one_turn_counts_once`, going through `_recorded` plus the end-of-turn logic or `AgentServer.handle`.
  - D3: the redirect sets `{"query": question.current}`.
  - D4: the rules are bullets 7-8 of `system_agent_case_f.txt`. Production runs `--hedge 0 --router-url` (`scripts/stack/up.sh start_agent`), and the hedged branch sends `messages` unsteered (`routed_turn`).
  - D9: `FIXED` maps "turn it up/down" to VOLUME_UP/DOWN. The smart_home handler appends `{"intent","slot"}` to the shared `record` on the fast path, the router path and the Gemini path.
  - `steer()` changes only the server-side copy of the last user message, and the client never receives it, so history and the system prompt stay byte-stable. The system string is built once in `run()`.
  - `venue.prompt_line()` returns None without a barangay. `venue.resolve()` runs in `run()` after the MCP children are spawned.
  - `dryrun.py` drives `Agent` directly, with no steer, fast path or Case F. So the dry-run tests only base-prompt behaviour, and the Now line can be covered only by unit tests.
  - Byte-exact steer tests exist in `test_prompt_scoping.py` (steer_*) and `test_chain_wiring.py` (active/idle chain). The opt-in `stamp` keeps them valid.
  - The size bound is currently 6876 and the prompt is 6803.
- **Must verify:**
  - The ported repro passes: 1 count per routed weather turn during `researching`, and the chain survives 2 such turns. It fails on HEAD.
  - The hedged branch carries the stamp. Use the existing hedged fakes in `test_agent_server_race.py`/`test_hedge.py` if they fit, else a minimal `HedgedAgent` with recording agents.
  - The system string passed to `build_llm` is identical across two turns. Assert that steering never touches it.
- **Unknowns:** whether flash-lite obeys the D9 sentence and the "never read it out" note. This is checked only by the optional dry-run and live use.
- **Dependencies/license:** none.
- **Symbols:** `agent_tools._recorded`, `one_fact_question`, `question.current`, `CHAIN`; `AgentServer.handle`, `steer`, `gemini`, `routed_turn`, `run`; `chain_state.touch_unrelated`, `is_chain_tool`, `SIDE_TOOLS`; `fast_path.FIXED`, `match`, `normalize`, `TIMEZONE`; `venue.current`, `prompt_line`.
- **Acceptance criteria:**
  - `test_one_turn_counts_once` passes; the old test is rewritten.
  - The redirect keeps "Who is Jensen Huang's wife?" when the model wrote it, and uses the transcript when `query` is empty.
  - The base prompt contains both D4 rules, and Case F no longer does. `test_no_rule_was_lost_in_the_split` still passes.
  - "Turn it up" after a TEMPERATURE/BRIGHTNESS record goes to the agent (RecordingAgent asked, no VOLUME_UP recorded, router not called). After a VOLUME/LIGHT/weather record, or with no history, it stays on the fast path.
  - With `stamp` set, every Gemini request's last user message ends with the `[Now: ... Place: ...]` line on the chain, no-router, escalated, D9 and hedged paths. The hedged request is otherwise identical to the unsteered one. Router input is unchanged.
  - The place falls back to the city when `venue.current.barangay` is None.
  - The "You cannot" block is present. The prompt size test passes with the new explicit bound.
  - The pipeline ruff, mypy and pytest gate is green.
- **Non-goals:** a router prompt change, Now lines for the router or fast path, `dryrun.py` changes, a `prompt_line()` city fallback, P4 annotations, a D11 notice fix, `TAVILY_COMMAND` (Task 3).
- **Security-sensitive:** no. Prompt text and routing logic only, with no secrets, new inputs or network.

### Task 3: pass the venue to the grounding child (Files: `pipeline/src/voice_pipeline/agent_tools.py`, `pipeline/tests/test_venue.py`)
- **Sequencing:** after T1, which produces the `--home` flag, and after T2, which shares `agent_tools.py`. It is small, so resume T2's developer.
- **Action:** replace the constant `TAVILY_COMMAND` with `tavily_command(env)`, mirroring `web_info_command`: `[..., "voice_agent.mcp_servers.tavily_search", "--home", env VOICE_AGENT_HOME_PLACE or "Cebu City"]`. Use it in `standard_servers`.
- **Established (verified):** MCP stdio children do not inherit the environment (`web_info_command` docstring; `stdio_server` passes `env` only for `MCP_PYTHON`). `web_info` already takes `--home`.
- **Must verify:** the interop check, run before review: `uv run python -m voice_agent.mcp_servers.tavily_search --help` from the repo root lists `--home`. This is free and makes no network call.
- **Unknowns:** none.
- **Dependencies/license:** none.
- **Symbols:** `agent_tools.TAVILY_COMMAND`, `web_info_command`, `standard_servers`.
- **Acceptance criteria:** a test like `test_the_venue_reaches_the_web_info_child_on_its_command_line` asserts `["--home", "Cebu City"]` by default and `"Mandaue"` when set. The pipeline gate is green.
- **Non-goals:** the barangay on the command line.
- **Security-sensitive:** no.

## Shared contracts
- **`tavily_search --home <place>`** (T1 produces, T3 consumes). There is one flag, and it has a precedent in `web_info`. This needs no `docs/` contract file; T3 verifies it with the interop command above.
- **`HEDGE`/source text vs `agent_tools.FAILURE`** (a cross-package contract). It is pinned by T1's literal-copy test, and the test-engineer adds the cross-check from the pipeline side if the pipeline venv can import `voice_agent`.
- **Fixture file schema** `{"id","query","system","force","captured_at","response"}`. T0 produces it, and T1 consumes the real files, never hand-built dicts.

## Test Boundaries
- **Task test owners (developer-owned):**
  - T0/T1: `tests/mcp_servers/test_gemini_grounding.py`, `test_tavily_search.py`. The fixtures come from the capture.
  - T2: `pipeline/tests/test_chain_wiring.py`, `test_prompt_scoping.py`, `test_venue.py`, `test_agent_server.py`/`test_turn_context.py`.
  - T3: `pipeline/tests/test_venue.py`.
  - There is no shared conftest work.
- **Test-engineer pass:**
  - An independent gap audit of the above.
  - Cross-cutting tests through the REAL entry points only:
    - the `web_answer` tool through `tavily_search.build_server(...)` and the fastmcp `Client.call_tool` with a fixture-replaying transport: the full chain zero-search -> forced -> Tavily -> hedge, cache behaviour across 2 calls, and the ledger rows written;
    - `AgentServer.handle` over a websocket (the `_turn` helper pattern): the stamp on the hedged and non-hedged paths, D2 across 2 routed side turns during `researching`, and the D9 routing.
  - Never reassemble the call graph from library functions.
  - Run both gates (root and pipeline).

## Risks
- **Latency on changeable zero-search answers:** about 0.9 s + 1.8 s = 2.7 s typical (today 0.9 s, but unverified). The worst case is about 4.5 s (Gemini budget) + 3.5 s (Tavily) = 8 s.
  - Mitigation: only the changeable class retries. The total grounded budget is 4.5 s. Settled and searched answers are byte- and latency-unchanged (tested).
- **The new prompt makes more questions search** (+0.9 s each, and search quota).
  - Mitigation: this is intended. `MONTHLY_SEARCH_CAP` of 4500 stays (151 used this month).
- **The classifier over-triggers** ("last", "new"). The cost is latency only, with no wrong answer.
- **The forced retry does not search.** The fixtures show it before T1 builds on it. If so, the fallback goes straight to Tavily (re-plan).
- **The hedge matches `FAILURE`** and causes a double lookup. A pinned test prevents it.
- **The Now line is read aloud:** a static "never read it out" sentence plus the same bracket convention as `[ACTIVE TASK]`, which works today.
- **Prompt growth of about +210 tokens per request, x2 when hedged:** about $0.00005 per request at $0.25/M. The bound is raised explicitly, not silently.
- **D2: cancelled (barged-in) turns no longer count toward expiry.** This is acceptable, and the old mid-turn count was the unintended one.
- **The ledger now holds user questions.** It stays in the local state dir, out of git and out of promote. Document it.
- **The D9 signal uses the process-wide `record`.** It is guarded by "the client sent an earlier assistant turn", so a stale record from an earlier session needs a fresh session's first turn, which has no history and therefore never triggers it.

- **Test pass: required.** Rationale: production behaviour, error-handling fallbacks, cache semantics, the ledger schema and prompt routing all change.
- **Documentation pass: required.** Rationale:
  - `docs/MCP-SERVERS.md` (web_answer): the rule, the gate, the source sentence, the hedge, the ledger columns and privacy, `capture_grounding.py` and its cost.
  - The front-agent doc (`docs/LIVE-AGENT.md` or `ARCHITECTURE.md`): the Now line, the cannot block, the D4 move, the D9 routing.
  - Mark D2/D3/D4/D9 closed where `docs/HANDOFF.md` lists them.
- **PR strategy:** not applicable. The remote `me3` is a local path with push disabled, and the owner said no push. Commits are by explicit file name only.

## Review-Finding Handling
R1/R2 findings are surfaced to the owner, and each retry waits for human approval (cap 2). Findings are never auto-fixed.

**WAITING FOR CONFIRMATION**
