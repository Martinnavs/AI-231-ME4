# Review: context-p1-p3 (tech-lead, 2026-11-14)

## Execution Log

### Review Feedback — R1 (uncommitted working tree on context-engineering, baseline 3969c3e)

**Scope.**
- Changed: `git diff` (15 files) plus the untracked `scripts/capture_grounding.py` and `pipeline/tests/test_turn_context.py`.
- No tickets exist; the orchestrator implemented T0-T3 directly. Judged against `PLAN.md`.

**Gates (run by the reviewer):**

| Gate | Result |
|---|---|
| root `pytest tests/mcp_servers` | 463 passed, 1 skipped |
| root full suite (known failure deselected) | 1389 passed, 14 skipped, 2 xfailed |
| root ruff check / ruff format / mypy | clean |
| pipeline pytest | 771 passed |
| pipeline ruff / format / mypy | clean |

**Known failure** `tests/doc_chat/test_flow.py::test_llm_echo_is_accepted_without_documents`: it fails with `ConnectionRefusedError` on 127.0.0.1:8765 (it needs a running speech server). `doc_chat` is untouched by the diff, so it is unrelated.

**Verdict:** the code is approved with no R1 and no code-defect R2. One R2 stays open and blocks closing P1: real fixtures are missing, and they need the owner's go to capture. Both workspaces' tests pass.

**Judged deviations:**
1. **D3 keeps the model's query only when the transcript matches `call_policy.REFERENCE`:** accepted. It keeps the existing `test_phrasing_stress` contract, and `REFERENCE` covers his/her/their/"that program". The redirect fires only on "who is ..." questions, so short follow-ups like "and AMD?" never take this path.
2. **`sources` + `lead` instead of a supports list:** accepted. It is simpler and the behaviour is the same.
3. **The Tavily source sentence only on the changeable fallback:** accepted. The plan's acceptance criterion asked only for that path, and other Tavily output stays unchanged.
4. **Inline fixture dicts instead of captured fixtures:** acceptable only as an interim state. It is finding CR-1 (R2) below. It also breaks the DESIGN.md edge "fixtures -> T1": T1 landed before its producer.
5. **The ledger "no question text" assertion reversed:** accepted. The owner required logging the query; the privacy note is in `docs/MCP-SERVERS.md`.
6. **No live calls, no dry run, docs written:** accepted as the state. The dry run and the live re-check remain the owner's choice.

**Checked and fine:**
- **Hedge caching:** `search()` returns `HEDGE` without `remember()`. Tested: the second call goes back to the network (4 Gemini requests).
- **Timeouts:** the first attempt is capped at 3.0 s. The retry gets `min(3.0, 4.5 - elapsed)` and is skipped when less than 1.0 s is left. A first-attempt timeout or error does not retry and falls to Tavily as before. The worst case is about 8 s. The router path's `LOOKUP_ACK_S` tool event moves the client's give-up cap to 20 s (`config/fillers.json`), so the turn is not dropped.
- **The stamp does not leak:**
  - The router gets the unstamped `user`/`history(messages)`.
  - `question.current` and the turn log use the raw text.
  - `Agent.ask` reads only `question.current`.
  - The client never receives the steered text.
  - The system string is built once.
- **Capture script:**
  - It reads the key only through `load_key()` and sends it only as the header.
  - It checks the key against the text before writing, and the refusal is tested.
  - Without `--yes` it makes no request (tested).
  - The `raise_for_status` text holds the URL, not the key.

**Findings:**

- `id: CR-1; severity: R2; primary_file/symbol: tests/mcp_servers/test_gemini_grounding.py::ZERO/FORCED (l.192-210); title: gate tests run on hand-built responses, not captured ones; impact: the P1 acceptance criterion "tests against tests/fixtures/grounding/*.json" is open, and the gate rests on unverified assumptions; owner_agent: test-engineer (after the owner approves the capture); confidence: high`
  - root_cause: capturing real responses needs live spend, which the owner has not yet approved.
  - file_paths: `tests/mcp_servers/test_gemini_grounding.py`, `src/voice_agent/mcp_servers/gemini_grounding.py` (`parse_sources`, `publisher`, `FORCE_SEARCH`), `src/voice_agent/mcp_servers/tavily_search.py` (`grounded`).
  - evidence: `ZERO`/`FORCED` are inline dicts. `tests/fixtures/grounding/` does not exist. These assumptions are unverified:
    - (a) the `--force` instruction actually produces `searches >= 1`;
    - (b) flash-lite returns `groundingSupports`;
    - (c) the real chunk titles are domains, so `publisher()` yields a sayable name;
    - (d) a real changeable zero-search response matches `ZERO`'s shape.
  - recommended_fix: after the owner approves, run the PLAN.md capture set (about 12-14 calls). Then:
    - load the fixture files in the gate tests (retry -> searched, retry -> Tavily, hedge, settled single-request);
    - assert `publisher()` gives a non-None, `FAILURE`-free name for every captured title;
    - assert that every `*_forced` fixture has `searches >= 1`. If one does not, escalate: the retry step is then useless, and the plan changes to going straight to Tavily.
- `id: CR-2; severity: R3; primary_file/symbol: pipeline/config/eval/system_agent.txt l.28 ("No, I mean ..."); title: in the base prompt, "make no new lookup" can block a corrected lookup; impact: "Who's the CEO of Nvidia?" -> "No, I mean AMD" may be answered from memory, with no source, which undercuts P1; owner_agent: developer; confidence: medium`
  - root_cause: a rule written for chain corrections was moved verbatim into the base prompt (the plan said verbatim; the reviewer missed this at planning).
  - recommended_fix: reword it to "redo that action once with the correction (a corrected lookup is that action); make no other lookup", and add the dry-run case ["Who's the CEO of Nvidia?", "No, I mean AMD."] expecting `web_answer` on both turns.
- `id: CR-3; severity: R3; primary_file/symbol: pipeline/src/voice_pipeline/agent_server.py::refers_to_setting l.271-279; title: "right after" is not enforced; impact: temperature -> a tool-less turn (a joke) -> "turn it up" (meaning music) goes to Gemini, which asks "to what temperature?", an extra question; owner_agent: developer; confidence: high`
  - root_cause: `self.record[before - 1]` is the last tool call ever made; turns without tools add nothing to it.
  - recommended_fix: remember `len(self.record)` at the start of the previous turn, and require the TEMPERATURE/BRIGHTNESS entry to fall within the previous turn. Add a test with a tool-less turn in between.
- `id: CR-4; severity: R3; primary_file/symbol: src/voice_agent/mcp_servers/gemini_grounding.py::publisher l.137 / HEDGE l.166; title: code-built spoken text can still match agent_tools.FAILURE; impact: a model guess that says "can't say" or "error", or a page title like "Home | Error" (publisher -> "Error"), makes the pipeline treat the reply as a failed lookup (stall line or a second lookup). The "Source: Error." case is also cached for 10 min; owner_agent: developer; confidence: medium (low likelihood)`
  - evidence: probed `publisher("Home | Error") == "Error"`. The pinned test checks only the literal `" Source: Nvidia."`.
  - recommended_fix: `publisher()` returns None for a name that matches the literal FAILURE copy. Drop the guess (hedge without it) when the guess matches. Extend the pinned test to cover `publisher()` output.
- R4:
  - The `grounded_research` ledger row has no `query` (`tavily_search.py` l.601).
  - `--legacy-prompt --force` silently drops `FORCE_SEARCH` (`capture_grounding.py` l.66).
  - The planned dry-run case ["Who painted the Mona Lisa?", "Got it, thanks."] is missing.
  - `publisher` gives "AbsCbn" and "Gmanetwork"; tune it against the captured titles.
  - No test covers the retry-skip path (less than `RETRY_MIN_S` left) or a retry that times out.
  - The new D4/D9 paragraph sits unlabeled between Case F and Case G.
- R5:
  - `test_one_routed_side_turn_counts_once...` manipulates `record.pop()`; it is correct but hard to read.
  - The test-engineer's independent pass from the plan did not happen. The implementer's fastmcp-`Client` and websocket `handle` tests cover most of the planned cross-cutting set.

**Needs from the owner:**
1. Approve the fixture capture (about $0.003-0.01 in tokens, 10-15 searches out of the 5,000 free, 151 used), and confirm the Gemini credit is live. That closes CR-1.
2. Optionally, the dry run of x01-x03 plus the CR-2 case (about $0.02-0.03) and the live re-check (about $0.01 plus up to 28 free searches).
3. Decide whether CR-2, CR-3 and CR-4 (non-blocking) are fixed now or deferred.
4. Confirm PLAN.md's defaults, which were implemented without an explicit confirmation: the source wording, hedge-with-guess, the 4.5 s budget, D9 asking, and the cannot list.
