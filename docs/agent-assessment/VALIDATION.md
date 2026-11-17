# Validation of the two independent assessments

Checked by Claude (Opus 5.5) on 2026-11-18 against the code at the `text-only` working tree, the turn logs in `pipeline/out/agent_server/`, `out/stack/logs/agent.log` and the ledgers in `~/.local/state/voice_agent/`. The reports are `CODEX-TERRA.md` and `AGY-ARGON.md` in this folder; the master document that adopts the confirmed findings is `../AGENT-ASSESSMENT.md` (problems F13-F21, comparison in section 13).

**Verdicts.** **Confirmed (run)**: reproduced by executing the code. **Confirmed**: verified by reading the cited code or data. **Partly**: the substance holds, a cited detail does not. **Wrong**: contradicted by the code or data. **Not checked**: left as the reviewer stated it; treat it as a lead.

## 1. Summary

| | codex (gpt-6.1-terra xhigh) | agy (gemini-4-argon-high) |
|---|---|---|
| Claims checked | 17 | 33 |
| Confirmed | 17 | 17 |
| Partly | 0 | 4 |
| Wrong | 0 | 12 |
| Wrong claims' type | none found | invented log lines and ledger fields, misquoted code and prompt text, wrong components, non-existent files, wrong mechanisms behind real effects |

Rule for reuse: codex's claims can be cited after a spot check; every agy claim needs checking first. Its real findings are among the most valuable of the exercise (F13 especially), so it is worth reading, not trusting.

## 2. codex: `CODEX-TERRA.md`

| # | Claim (report section) | Verdict | Evidence |
|---|---|---|---|
| C1 | Hedged turns get only the timestamp tail, not `steer`'s changeable note or Case F (§1, §3.1) | Confirmed | `agent_server.py:587-598` uses `stamped()` (`:264-272`); `steer()` (`:241-262`) only at `:609, 819, 830` |
| C2 | An unsupported answer is logged `unchecked`, not blocked; the test asserts logging (§3.1) | Confirmed | `agent_server.py:741-744, 875-884`; `pipeline/tests/test_changeable.py:100-106` |
| C3 | `doc_answer` accepts "Alice is not the CEO" against the quote "Alice is the CEO" (§3.2) | **Confirmed (run)** | `judge()` returned `accepted` (section 4) |
| C4 | `passages_of` drops the source; a document answer cannot name its file (§3.2, D4, S9) | Confirmed | `doc_answer.py:75-93` returns texts only |
| C5 | Case C answers bypass the document checker (§3.2) | Confirmed | `system_agent.txt:20`; the checker runs only in `doc_race` (`agent_server.py:336-362`) |
| C6 | Per-turn lookup reuse is not keyed by query or tool (§3.4) | Confirmed | `agent_tools.py:1054-1070` returns `_turn_lookups[0]` to any later lookup |
| C7 | The cache key drops "who, is, was, current, now"; "who is" and "who was the CEO of Nvidia" collide (§3.4) | **Confirmed (run)** | both give `{ceo, nvidia}`, `same_question` True (`tavily_search.py:87-95, 243-254`) |
| C8 | The corpus loader reads only `.md`/`.txt`, skips unreadable/empty files silently, keeps only the basename (§3.5) | Confirmed | `src/doc_chat/corpus.py:30-47` (and it exits the process on an empty folder, `:26-33, 49-51, 58-60`) |
| C9 | Five early second looks are SUPPORTED with zero searches; later ones search (§3.6) | Confirmed | `gate_outcomes.jsonl` rows 1-5 (0 searches), 9-11 (1-2) |
| C10 | The ledger has about 226 zero-search rows of 414 (§5) | Confirmed (≈) | 230 of 418 on 2026-11-18 07:24 (the ledger grew) |
| C11 | A correction notice is shown but not added to the next turn's history (§1 Memory) | Confirmed | `src/text_chat/cli.py:63-82` |
| C12 | The dry run collapses repeated calls to one tool, so it cannot see repeated searches (§5) | Confirmed | `dryrun.py:145-146` ("repeated searches in one turn count once") |
| C13 | `search_eval.py` has 8 cases that check required words (§5) | Confirmed | `config/eval/search_cases.jsonl` (8 lines, `must` keywords) |
| C14 | The Tavily hedge still speaks "the closest result says: ..." (§2, E2) | Confirmed | `tavily_search.py:283-286, 711-716` |
| C15 | Three runs gave three different "latest KoolPals episode" answers (§3.3) | Confirmed | 1032 and 1053 (`20261116-104540.jsonl`), 1056 (`20261116-120321.jsonl`) |
| C16 | A research answer stated an allegation about a named person with no source in the sentence (§3.3, S8) | Confirmed | `20261116-104540.jsonl` row 8 ("after he admitted to extramarital affairs ...") |
| C17 | The routed document race prefers the document answer and cancels the web task; it never compares them (§1, D5) | Confirmed | `agent_server.py:388-405` |

Not checked (codex): the router grammar and deadline details (§2), the stall-fallback behaviour (§2), the description of `chain_eval.py` checks (§5), and the inferred, untested risks it labelled as such.

## 3. agy: `AGY-ARGON.md`

| # | Claim (report section) | Verdict | Evidence |
|---|---|---|---|
| A1 | The client talks over a UNIX socket with an HMAC challenge; the online probe is TCP to 8.8.8.8 (§1.1) | Wrong | WebSocket with a token header (`agent_server.py:116-132`); the probe host is `generativelanguage.googleapis.com` (`:140`) |
| A2 | The router is a 0.5B FunctionGemma (§1.1) | Wrong | Granite 4.0 H 350M with a LoRA adapter (`README.md`, "The router model") |
| A3 | A stale research chain from an earlier session steered a new server's first turns and bypassed the router (P1) | **Confirmed** | `chain_update` KoolPals at 12:03:43 (`20261116-120321.jsonl`); server restart 12:07:40; first turn sent with `[ACTIVE TASK: phase=researching ...]` (`agent.log:240-241`); no `router_lines` in `20261116-120744.jsonl`. Adopted as F13 |
| A4 | ... because the losing hedged request also ran `chain_update` and overwrote the chain back (P1) | Wrong | `gated_tools` runs handlers for the winner only (`hedge.py:65-92`); only `GoogleLLMService#1` called `chain_update` (`agent.log:265-266, 305-306`) |
| A5 | `chain.json` lives 1,800 s (P1) | Wrong | `EXPIRE_S = 600.0` (`chain_state.py:26`) |
| A6 | Per-turn lookup reuse is query- and tool-blind (P2) | Confirmed | `agent_tools.py:1054-1070` |
| A7 | A failed first lookup resolves the shared future with its failure text, and a waiter returns "(already looked up this turn) <failure>" (P2) | Confirmed | `agent_tools.py:1094-1096` sets the failure text; `:1063-1070` returns any non-empty value |
| A8 | `changeable_question` needs a question word at the very start; "hello, can you tell me ..." fails (P3) | Confirmed | `call_policy.py:650-661` (`_ASKS.match`, no "hello", no "can") |
| A9 | ... so the second look was not scheduled for that turn (P3) | Wrong | the second look depends on `turn_claim()`, not on `changeable_question` (`agent_server.py:868-869`) |
| A10 | The changeable note is missing on hedged turns (P3) | Partly | the effect is confirmed (C1); the note text it quotes ("[Note: this question asks about a fact ...]") is not the real text (`call_policy.py:668-671`) |
| A11 | Tavily `web_research` results carry no source marks, so the backstop cannot cite them; the backstop is skipped when the reply contains "source" (P4) | Confirmed | `tavily_search.py:196-216`; `agent_tools.py:366-403` (`"source" in said`) |
| A12 | Tavily's `web_answer` adds the source only when a grounded guess existed (P4) | Confirmed | `tavily_search.py:713-714` |
| A13 | Rows 1, 8, 12 name no publisher; row 9 says "sources including" without naming any (P4) | Partly | row 9 names The Guardian and ABS-CBN (`20261116-104540.jsonl`); the suppression mechanism (A11) is real |
| A14 | `corpus.py` exits the process on a missing or empty folder; `.md`/`.txt` only (P5) | Confirmed | `corpus.py:26-33, 49-51, 58-60` |
| A15 | The document topic is hard-coded to baby-onit / onit (P5) | Confirmed | `call_policy.py:419`, `docs.py:18-21`, `system_agent.txt:20` |
| A16 | `names_numbers_ok` matches digits as substrings ("5" in "450") (P5) | **Confirmed (run)** | "Acme earned 5 million." against "450 million" was `accepted` (section 4) |
| A17 | "158,000" does not match "158000" (P5, S11) | **Confirmed (run)** | `check_failed` (section 4) |
| A18 | `search_docs` results have no untrusted-data wrapper (P5) | Confirmed | `docs.py:40-46` |
| A19 | `web_verify` is off by default; in the 10:45 session every check was skipped (P6) | Partly | the tool's default is 0 and the 10:45 session logged 4 × `SKIP: checks are off`; `make text-up` sets 100 (`docs/TEXT-MODE.md`, "Second opinions") |
| A20 | `turn_claim` checks the tool's output, not the spoken text; notices are shown only after the next input (P6) | Confirmed | `agent_tools.py:406-412`; `cli.py:63-68` |
| A21 | `gate_outcomes.jsonl` line 4 is a DIFFERS verdict on the Joe Rogan episode (P6) | Wrong | line 4 is SUPPORTED, "Who won the last NBA championship?"; the file has no DIFFERS row |
| A22 | Ledger rows have `changeable`, `grounded`, `queries`; 119 rows are `grounded: false`; 7-day TTL; line 1 is the swallow question; the model is Gemini 2.5 Flash (P7) | Wrong | rows hold `time, month, searches, tokens, seconds` (+ `query, sources, forced, check`); line 1 is a 2026-11-03 row with no query; answer cache 600 s (`tavily_search.py:74`); model `gemini-3.1-flash-lite` (`gemini_grounding.py:34`) |
| A23 | The Tavily "closest result" hedge is spoken and cached (P7) | Confirmed | `tavily_search.py:283-286, 711-716` |
| A24 | The prompt line 28 says "accept the correction ... make no new lookup"; `agent_tools.py:1029` clears lookups on `is_repair` (P8) | Partly | the CR-2 risk is real (`.scratch/context-p1-p3/REVIEW.md`); the quoted prompt wording is not the file's (`system_agent.txt:28`), and there is no `is_repair` |
| A25 | `self.record` holds the user and assistant messages (P9) | Wrong | it holds tool calls (`agent_tools.py:1046-1052`, `agent_server.py:201`); the substance (history lacks tool evidence) is D6 of `docs/CONTEXT-CRITIQUE.md` |
| A26 | `FAILURE` is `\b(could not|couldn't|failed|unable to|error|unavailable|cannot|can't)\b`; failed calls become `"FAILED: <tool>"`; this erased calls in rows 9 and 11 (P10, §5.2) | Wrong | `FAILURE` is `couldn't find|did not answer|not connected|can't say|error` (`agent_tools.py:185`); no "FAILED:" exists; row 9 lists `web_research`, row 11 is the Galileo question with `web_answer` |
| A27 | `run_bash` always enables sandbox network via `allow_network=True` / `_build_bwrap_cmd`; `subagent.py:446` flags "error" substrings (P11) | Wrong as cited | those symbols do not exist; the runner shares the network only when `policy.network` (`runner.py:32, 185-186`). Which policy `run_bash` gets is **not checked** |
| A28 | Usage metrics are off (P12) | Confirmed | `agent.py:373`, `agent_server.py:980, 987` |
| A29 | ... the missing usage is OpenRouter's (P12) | Wrong | the front model is Gemini through Pipecat's Google service |
| A30 | `agent_prompts.jsonl` has duplicate ids (s01, s02, m01, t01-t03, x01-x03) (§5.2) | Confirmed | 9 duplicated ids in 90 rows |
| A31 | Prompts expect tools hidden from the model (`convert_time` t03, `wiki_summary` k01-k03) and x01 asks about "The Road", which is not in the corpus (§5.2) | Confirmed | `agent_tools.py:823` (`MODEL_HIDDEN`); `grep Road corpus/` finds nothing |
| A32 | `output_audit.py` hard-codes the ME3 log folders (§5.2) | Confirmed | `scripts/output_audit.py:29-31` |
| A33 | `output_audit.py` checks `unattributed_search`, `markdown_leak` etc. with a regex that "from" satisfies; a test skips "no captured fixtures yet"; 9 search cases and 24 chain scenarios with `must_lead_as_per` fields; reading files `M3_advanced-rag-techniques.md`, `M4_domain-specific-rag-and-trusted-generation.md` (§5) | Wrong | `output_audit.py` has 4 checks (`:41-46`); no such skip in `tests/mcp_servers/test_gemini_grounding.py`; 8 and 23 cases with `must`/`keywords`; those reading files do not exist |

Not checked (agy): its §2 guard inventory line by line, its R1-R9 code sketches, its latency numbers, its test-file counts, and which sandbox policy `run_bash` uses.

## 4. Commands run for the "Confirmed (run)" rows

```python
# pipeline venv: voice_pipeline.doc_answer.judge on one passage
P = ["The board said Alice is the CEO of Acme since 2020. Revenue was 450 million and 158000 units shipped."]
judge("ANSWER: Alice is not the CEO of Acme.\nQUOTE: Alice is the CEO of Acme", P).status  # accepted
judge("ANSWER: Acme earned 5 million.\nQUOTE: Revenue was 450 million", P).status          # accepted
judge("ANSWER: Acme shipped 158,000 units.\nQUOTE: 158000 units shipped", P).status         # check_failed

# root venv: voice_agent.mcp_servers.tavily_search
content_words("Who is the CEO of Nvidia?") == content_words("Who was the CEO of Nvidia?") == {"ceo", "nvidia"}
same_question(...)  # True
```

These are ready-made regression tests for F15 and F16: each should flip when the fix lands.
