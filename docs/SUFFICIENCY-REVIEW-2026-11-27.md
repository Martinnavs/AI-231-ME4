# Sufficiency Check Adversarial Code Review (2026-11-27)

Post-deployment review of `exp/sufficiency` (`sufficiency.py`, `_maybe_supplement` in `agent_server.py`, and interactions with `ask_documents.py`, `compound.py`, `agent_tools.py`, `spoken_filter.py`, `tavily_search.py`, and `src/doc_chat/`). **Do not enable `SUFFICIENCY=True` until all four R1 blockers and six R2 issues are fixed.**

## Summary Verdict

| Severity | Count | Key Themes |
|---|---:|---|
| **R1 (blocks enabling)** | 4 | `REPLY_SCHEMA` breaks production `verdict()`, `ask_documents(status="web")` double-lookup + `REPEAT_PREFIX` spoken aloud, `wrong_question` wastes search + poisons `turn_source_sentence`/`web_verify`, disconnect/cancel leaks task & stalls `self.lock` |
| **R2 (fix before enabling)** | 6 | Pure-audio first-audio IS delayed + `SentenceFilter`/space/40-word bugs, `FY` stripped before `source_earlier`, gate under-triggers `missing_half`/short answers, missing `offline()` check + shielded timeout billing, late-audio dead air, PII in `web_question` |
| **R3 (later)** | 2 | `"newer figure"` wording on dates/people/rules; JSON fence stripping & `today_line()` timezone |

---

## R1 Findings (Blocks Enabling)

### R1-1: Production `gemini_json_request` enforces `REPLY_SCHEMA`, making `verdict()` always return `None`
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:1947`, `pipeline/src/voice_pipeline/ask_documents.py:134-145,1232`, `pipeline/src/voice_pipeline/sufficiency.py:182-194`
- **Scenario:** Production wires `doc_json_request = await gemini_json_request(...)`, which hardcodes `response_json_schema=REPLY_SCHEMA` (`required: ["status"]`, properties `status/answer/file/quotes/calc/web_question`). `sufficiency.verdict` requires `{"verdict", "web_question", "reason"}`. Gemini's structured output cannot emit `"verdict"`, so `v not in VERDICTS` (`sufficiency.py:192`) always triggers and returns `None` after billing the API call. `test_predeploy_250.py:103` masked this with a schema-ignoring fake. (Same bug affects `compound.py:87-113`.)
- **Fix:** Pass a `response_schema` argument into `JsonRequest` / `gemini_json_request` (or build a dedicated requester with `SUFFICIENCY_SCHEMA` requiring `reason`, `verdict`, `web_question`).

### R1-2: `ask_documents` `status=="web"` triggers double lookup and speaks `"(already looked up this turn) "`
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:808-858,1427,1514`, `pipeline/src/voice_pipeline/ask_documents.py:1112-1128`, `pipeline/src/voice_pipeline/agent_tools.py:1284-1328`
- **Scenario:** When `ask_documents` returns `status="web"`, it has already called `web_answer` (`_web(web_q)`) and appended `web_text` to `doc_result.spoken`, yet still returns `status="accepted"`. Both hook sites call `_maybe_supplement` without checking `doc_result.web_text` or `doc_result.web_question`. If `should_check` passes, `_maybe_supplement` calls `verdict` and `self.lookup("web_answer", ...)` again. In production (`_recorded`), `_turn_lookups` is already populated, so `agent_tools.py:1286-1289` skips the web search and returns `"(already looked up this turn) <first_web_reply>"`. `_maybe_supplement` (`:851-857`) does not filter `REPEAT_PREFIX`, speaking the internal marker aloud and adding a third part to the turn.
- **Fix:** Return `await speak` immediately when `doc_result.web_question` or `doc_result.web_text` is non-empty; guard against `REPEAT_PREFIX` in `_maybe_supplement`.

### R1-3: `wrong_question` pays for `web_answer`, discards it, and triggers spurious citations & `web_verify`
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:836-850,1744,1761`, `pipeline/src/voice_pipeline/sufficiency.py:121-122,141-143`, `pipeline/src/voice_pipeline/agent_tools.py:455-485,1318`
- **Scenario:** `sufficiency.SYSTEM:141-143` says `web_question` is empty for `wrong_question`, yet `agent_server.py:836` runs `self.lookup("web_answer", ...)` for `"wrong_question"`, logs only `got.web_question` (`:850`), and drops `web`. In production (`_recorded`), `_turn_lookups` records the unspoken web reply. At turn end (`:1744,1761`), `turn_source_sentence` sees the web publisher is absent from `result["text"]` and appends `"Source: <Publisher>."` to the unrelated document answer, and `turn_claim()` fires `check_in_background` (`web_verify`) against that discarded web answer—spending extra search/compare calls and risking a bogus unacked notice.
- **Fix:** Exclude `"wrong_question"` from the `web_answer` call at `agent_server.py:836` (log `got.verdict`/`got.reason` only); isolate supplement lookups from `turn_claim()` or pass the actual spoken claim.

### R1-4: WS disconnect / cancellation does not stop `_maybe_supplement`, leaks task, and blocks `self.lock`
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:816-845,1287,1719-1725`
- **Scenario:** When a user barges in or disconnects, `ws.wait_closed()` finishes first at `:1721`. `handle()` only calls `self.agent.cancel()` (which does not cancel `turn`) and then `await turn` (`:1725`) while still holding `self.lock` (`:1287`). `_maybe_supplement` can keep running `verdict` (up to `AWAIT_TIMEOUT_S=2.5s`) and `web_answer` (up to `WEB_TIMEOUT_S=4.0s`), stalling the next turn (even fast ME3 commands) for up to 6.5 s and spending web calls after hangup. Separately, if `speak` (`:825`) is cancelled (`CancelledError`) or raises, `task` (`:816`) has no `try/finally` cancellation and leaks past the turn.
- **Fix:** Cancel `task` in `try/finally` inside `_maybe_supplement`; cancel `turn` (or check a turn-cancelled event before `web_answer`) when `closed` wins `asyncio.wait` in `handle()`.

---

## R2 Findings (Fix Before Enabling)

### R2-1: Pure-audio first audio IS delayed; `SentenceFilter`, missing space, and 40-word cap break supplements
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:853-857,1264-1274,1732`, `pipeline/src/voice_pipeline/spoken_filter.py:105-107,155-180`, `src/doc_chat/chain.py:270-276,370`, `src/doc_chat/__main__.py:1701`
- **Scenario:** (a) `on_text` sends `{"text": ...}` without `"flush": true`. `SentenceStream._sentences(..., final=False)` holds the single-sentence `doc_result.spoken` in `last[:-1]` until either a second sentence arrives or `{"done": true}` (`final=True`)—delaying TTS first audio up to 6.5 s (`test_predeploy_250.py:430` only measured `on_text`). (b) `await on_text(lead + text)` sends no leading space. When `SentenceFilter.passing` is `True` (`"As"` not in `_OPENERS`) and `doc_result.spoken` ends with `"."` (no space), `SentenceFilter.feed` emits `"As per sheet.csv: 12.5.The web has..."` with no space, breaking `split_sentences` (`src/lan_speech/protocol.py:312`). (c) `"the"` is in `_OPENERS`: after a multi-sentence doc answer (`passing=False`), `"The web has a newer figure: <1 sentence>"` is held in `sentences.held` until `sentences.finish()` (`:1732`). (d) `SentenceStream._take` drops sentences exceeding `--max-words 40`, silently truncating the supplement on TTS while `agent_server` logs it as spoken.
- **Fix:** Flush `doc_result.spoken` (`flush: true`), normalize trailing `.` + leading space before `on_text`, bypass `_OPENERS` holding for server supplements, and exempt flushed supplements from the 40-word client cap (or bump cap for multi-part turns).

### R2-2: `temporal.source_year` strips `"FY"` before `source_earlier` and ignores `card.date_max`
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:813`, `pipeline/src/voice_pipeline/temporal.py:19-26`, `pipeline/src/voice_pipeline/sufficiency.py:59,85-103`
- **Scenario:** `agent_server.py:813` passes `date_max=None` to `temporal.source_year`, which only matches `20\d{2}`. For `fy2026_per_diem_master_rates.xlsx` (`imp-001`), it returns `"2026"` instead of `"FY2026"`, so `source_earlier("2026", 2026-11-27)` returns `False` (`2026 < 2026`), making `_FISCAL` dead code even though FY26 ended in September. `fy26_rates.xlsx` returns `None`; `nst_est2024_popchg_2020_2024.csv` returns `"2020"`; and `source_earlier("2025")` over-triggers on timeless 2025 rules (`imp-049`).
- **Fix:** Detect `FY\d{2,4}` in `temporal.source_year` (or pass `doc_result.file` directly to `source_earlier`), wire `file_cards.date_max` into `DocResult`, and avoid `source_earlier` triggers when the question asks about a past/fixed rule.

### R2-3: Gate under-triggers on `missing_half`, `format_citation` pads word count past `SHORT_ANSWER_WORDS`, and cues drift
- **Where:** `pipeline/src/voice_pipeline/sufficiency.py:41,54-78,105-116`, `pipeline/src/voice_pipeline/agent_server.py:814`
- **Scenario:** (a) `:814` passes `doc_result.spoken` (`"As per <file>: <answer>"`, +3–4 words) instead of `doc_result.answer`, so `len(words) < 6` only fires on 1–2 word answers (`test_predeploy_250.py:218` used a 5-word string without `"As per"`). (b) On `eval/sufficiency/implicit_compound.jsonl`, `needs_web_missing_half` items with current/undated files (`imp-022,027,028,031,033,035`) lack time keywords and `<2026` years (`1993` in `imp-028` ignored by `20\d{2}`), so >=3-word answers skip `should_check`. (c) `_TIME_SENSITIVE` misses `"last month"`, `"next fall"`, `"next January"`, `"this fiscal year"`, `"this semester"`, `"so far"` (`imp-004,011,012,015,017,037,038`), while `_grounding.changeable` over-triggers on static `"price"/"rate"/"last"/"new"`.
- **Fix:** Evaluate `SHORT_ANSWER_WORDS` on `doc_result.answer or ""`, expand `_TIME_SENSITIVE` to match `temporal`/`_NEXT_YEAR_CUE_RE` + `"last month"`/`"this semester"`/`"next <season/month>"`, and measure `missing_half` recall in Experiment 1.

### R2-4: Missing `self.offline()` guard; `WEB_TIMEOUT_S` + `asyncio.shield` bills ledger after timeout
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:472-477,839-846`, `src/voice_agent/mcp_servers/tavily_search.py:427-431`
- **Scenario:** `_maybe_supplement` calls `self.lookup("web_answer", ...)` without checking `self.offline()` (unlike `acked_call:445` and `routed_turn:1063`), stalling 4.0 s on `WEB_TIMEOUT_S` + transient retry when offline. Online, `tavily_search.py:427` shields the MCP task, so a timed-out 4.0 s lookup still completes and appends to the spend ledger.
- **Fix:** Skip `web_answer` when `self.offline()`; use a single non-retried lookup with a `<= 2.5s` timeout.

### R2-5: Pure-audio late-arrival UX (up to 6.5 s dead air without tool ack/filler)
- **Where:** `pipeline/src/voice_pipeline/agent_server.py:825-846`, `src/doc_chat/__main__.py:610-640`
- **Scenario:** Even after fixing `flush: true`, `self.lookup` emits no `{"event": "tool"}` progress frame, leaving up to 6.5 s of silence after the doc answer finishes before the supplement suddenly plays or collides with the user's next utterance.
- **Fix:** Enforce a tight post-answer deadline (e.g., `<= 1.0s` remaining for verdict after `speak`, `<= 2.5s` for web; skip supplement silently if exceeded or if `ws` closed).

### R2-6: Privacy leakage from private documents into public `web_question` queries
- **Where:** `pipeline/src/voice_pipeline/sufficiency.py:130-156,175-180`, `pipeline/src/voice_pipeline/agent_server.py:841-844`
- **Scenario:** `sufficiency.SYSTEM` does not instruct the verdict model to exclude private names, amounts, or internal identifiers from `got.web_question` before sending it to public Google Search / Tavily (`web_answer`). `Spoken answer` is also not marked untrusted data.
- **Fix:** Instruct `SYSTEM` to strip private/personal/file details from `web_question` (public concepts only) and treat `Spoken answer` as untrusted data.

---

## R3 Findings & Interaction / ME3 Checks

- **R3-1 (`agent_server.py:853-855`):** `"The web has a newer figure: "` fires for stale dates, people, and policies (`imp-012,036,039`), and stacks awkwardly with `"As per <Publisher>, ..."`. Use `"More recently, "` / `"From the web: "` and ensure addition-only framing.
- **R3-2 (`sufficiency.py:159-161,186`):** `json.loads(raw)` fails on markdown fences (unlike `gemini_grounding.parse_directness:334`), and `today_line()` uses host local time instead of `ZoneInfo(fast_path.TIMEZONE)`.
- **245 / 247 / 250 interaction & ME3 regression:** `247` (`compound.answer`, `_docs_miss_turn`) and `250` (`_maybe_supplement`, `accepted`) are mutually exclusive via `handle()`. However, `245` (`DIRECTNESS` in `tavily_search.py:570`) runs on every `web_answer` call, so `ask_documents(status="web")` + `245` + `250` fire together and speak three things (R1-2). Hedge handling (`HEDGE_TAIL not in text`, `:851`) correctly suppresses hedged web replies, though `lookup()` failures return `None`. With `SUFFICIENCY=False`, ME3 `fast`/`routed`/`chain`/`gemini` paths are unchanged byte-for-byte; when `SUFFICIENCY=True`, R1-4 can stall ME3 commands behind `self.lock` for 6.5 s after a barged-in docs turn, and `241`'s broad docs routing can send misrouted turns into `_maybe_supplement` if `ask_documents` accepts them.
