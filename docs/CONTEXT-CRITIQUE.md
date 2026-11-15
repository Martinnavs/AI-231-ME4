# Context handling critique (2026-11-14, branch context-engineering)

Scope: how the front agent's context is injected, carried between turns, kept across sessions and checked, judged against OnIt (the reference clone) and the professor's five rules. Evidence comes from reading the code at `3969c3e`, the grounding ledger (`~/.local/state/voice_agent/web_answer_ledger.jsonl`, 350 rows, 2026-11-03 to 11-12), the ME3 agent-server logs (253 turns) and one scratch reproduction (D2). No code was changed.

## 1. Bottom line

1. Weak on rules 1, 4 and 5 for web facts. 208 of 348 grounded `web_answer` calls (60 %) ran **zero** searches. The answer is then spoken as it is, with no source, no check, and a 10-minute cache. The CEO failure is the designed behaviour, not a fluke.
2. Rule 3 is half done. The default prompt never gets the date (`{today}` exists only in the builtin-search prompt). The location appears only when the barangay lookup succeeded at boot. "Cannot do" is implicit. What we already inject (owner, venue, phonebook, Case A–G) is decent and cache-stable.
3. Within one session, context is a fixed window: the last 4 spoken pairs for Gemini and 1 for the router. It holds no tool calls and no sources, has no summary, and nothing comes back once a turn falls out. Sessions run 5–89 turns (median about 27), so a fallen-out turn is common. The chain state is the only structured memory, and it has a reproduced double-count bug (D2).
4. Across sessions we keep nothing. OnIt has episodic recall, but it ships **off** by default (autonomy `observe`). Neither has thread tracking. The gap to OnIt is large on verification and moderate on memory. On per-turn latency engineering we are ahead (router, fast path, doc check).
5. Cheapest big win: make grounded answers prove a search and carry a source (a deterministic gate, no extra LLM call). Second: fix four small context defects. Thread tracking and episodic memory are worth building only after a labelled fixture shows how often threads return.

## 2. MECE table

Gap: L = large, M = medium, S = small, 0 = none. Claim column: whether your preliminary claim held. **Corrected** marks a claim I had to change.

| Axis | OnIt | Ours (evidence) | Gap | Your claim |
|---|---|---|---|---|
| **Inject: date/time** | `{current_date}` in the volatile half of every instruction (`prompts.py:162,207`) | Only when `--builtin-search` (`agent_server.py:870-873`). `system_agent.txt` has no `{today}` placeholder. The front agent has `get_current_time` as a tool, so rule 3's "can be gotten during the conversation" is met, but Case G reasoning ("days until Christmas") has no date. | M | Held |
| **Inject: location** | none (CLI) | `venue.prompt_line()` returns None unless the barangay resolved (`venue.py:105-114`), so offline at boot means no place line at all, not even "Cebu City". Time zone hard-coded `Asia/Manila` (`system_agent.txt:15`, `fast_path.py:20`). | S | (new) |
| **Inject: grounded sub-prompt** | n/a (the main model searches itself) | `gemini_grounding.SYSTEM` gets the date, not the venue (`gemini_grounding.py:46-52`) | S | Held |
| **Inject: can / cannot / format** | static rules incl. an explicit "You cannot install anything" block (`prompts.py:290`), References section required (`:402`), untrusted-content rule (`:410`) | Can: Case A–G (`system_agent.txt`). Format: spoken, 1–3 sentences. Cannot: implicit (scattered "never" lines). Untrusted-data header only in NOTES.md for the delegate (`scratchpad.py:18`). | S–M | Held |
| **Inject: metadata (owner, sandbox, files)** | working dir, sandbox, file server (`prompts.py:205-212, 256-263`) | Owner name and profile rule (`agent_server.py:846-854`), delegate sandbox described in Case F, apps listed on demand (`open_app`) | 0–S | (new) |
| **Inject: rule-1 wording** | "Anything that can change needs a tool result, not memory: … who holds a job" (`prompts.py:364`) | Grounder: "Do not search for settled facts", thinking `minimal` (`gemini_grounding.py:51,124`). The model decides that a CEO is settled. | **L** | Held |
| **Flow: history window** | 6 replayed turns, answers older than 3 cut to 800 chars (`chat.py:2870-2900`, `onit.py:895`) plus in-run compaction at 0.5–0.9 of the window | Client keeps 4 pairs (`talk.sh:25 --history 4`, `dialogue.py:42-53`). Server resends them with `reset=True` (`agent.py:418`). Router sees 1 pair (`router.py:29`), with Gemini replies cut to 100 chars (`agent_server.py:233`). | M | **Corrected**: the window is client-side `--history 4`; `HISTORY_PAIRS=1` is the router's only |
| **Flow: what history contains** | task and response; RunState "prior attempts" lists the tools run (`state.py`, `prompts.py:507-513`) | Spoken text only. No tool calls, args or sources. Barge-in stores `"(interrupted)"` or the partial words (`__main__.py:1234`). A played notice takes a slot as a fake user turn `"(a background job finished)"` (`__main__.py:675`). | M | (new) |
| **Flow: compaction / summary** | yes (in-run), and the model is told (`harness.py:248-255`) | none | S at 4 pairs (nothing to compact); the real loss is turns 5+ dropping out | Held, but low value (see section 4) |
| **Flow: per-tool result budgets** | central preview + handle + `result_read` (`results.py:60`, 6000 chars) | Per tool, ad hoc: `web_answer` 2 sentences/400 chars (`tavily_search.py:73-74`), `web_research` 2000 (`:105`), file read 1500 (`agent_tools.py:549`), docs top-3 (`docs.py:30`), chain facts 700 (`chain_state.py:31`) | S | **Refuted**: budgets exist; there is no central clamp |
| **Flow: prompt ordering / cache** | static/volatile split, playbook pinned at the end of the static half (`prompts.py:519-575`) | Mostly right by accident: a static system prompt per process; chain block and Case F protocol appended to the **last user message** (`agent_server.py:238-255`). Cached tokens are never measured (`Agent(usage=False)`, `agent.py:373,383`). | S | (new) |
| **Retain: model-callable memory** | `context_status`, `note_write`/`note_read` (`harness.py:100-130, 420-484`) | none (`chain_update` is task state, model-writable) | S for us (see P8) | Held |
| **Retain: cross-session** | `learn/recall.py`: BM25 over episodes, owner and model filters, outcome bands, 800-token cap, untrusted marker. **Default off** (`learn/config.py:47`, gated `onit.py:1660`). | Owner profile (`config/profile.md`), cost ledgers, server turn logs (no date, no session id, no sources: `agent_server.py:789-797`). Nothing is read back. | M–L | Held (noting OnIt's is off by default) |
| **Retain: notes / scratch** | per-run notes on disk | `NOTES.md` TTL 600 s, 12 KB, read **only by the delegate**, never by the front agent (`scratchpad.py:20-24`) | S | Held, plus: the front agent never sees it |
| **Retain: thread-like state** | none | `chain_state`: ends after 600 s idle, 3 unrelated turns, or a non-follow-up after done (`chain_state.py:26-27`, `agent_server.py:719-727`). The unrelated count is double-counted (D2). | M | Held, plus a bug |
| **Verify: post-answer check** | `verify.py`: a fast verdict against tool evidence, hard deadline, fails open; default on, but skipped under 80 chars (`verify.py:129-136`), so it would **not** fire on "Jensen Huang is the CEO" | Grounded path: none (`tavily_search.py:545-547` says so). Tavily path: `unconfirmed()` name/number check and hedge (`:250-288`). Docs: verbatim-quote check (`doc_answer.py`). Delegate: keep-all check. | **L** for web facts | Held for web; **corrected** that checks exist elsewhere |
| **Verify: source returned** | References section required | Grounding titles parsed then dropped for `web_answer` (`tavily_search.py:548`); Tavily titles/URLs dropped (`spoken()`, `:224-234`); `SourceFilter`/`strip_sources` remove "(Source: …)" asides (`payload_guard.py:62,108`) | **L** | Held |
| **Verify: low-confidence → clarify** | "ask exactly one clarifying question" (`prompts.py:498`) | Case E (home) and a Case F paragraph; grounded answers have no "say you don't know" path | M | (new) |

## 3. Defects, ranked by severity

**D1 (high). Unsearched answers are spoken as fact, with no source, and cached.** `one_fact_question()` (`agent_tools.py:946-952`) correctly sends "Who's the CEO of Nvidia?" to `web_answer`. That calls `gemini_grounding.ask`, whose prompt says not to search settled facts, with thinking `minimal`. The ledger shows `searches: 0` on 60 % of calls (median 0.91 s, against 1.8 s with one search). `grounded()` caches the reply for 600 s keyed by content words (`tavily_search.py:549, 68`). `web_answer` is in `SPOKEN_RESULT_TOOLS` (`agent_tools.py:160-170`), so no second model pass sees it. The ledger has no query column, so I **cannot** say how many of the 208 were changeable questions. That is the first thing to log.

**D2 (high, reproduced). An active chain counts one unrelated turn twice.** `_recorded` calls `CHAIN.touch_unrelated()` for every non-chain tool call (`agent_tools.py:983`). `AgentServer.handle` calls it again at the end of the turn when no chain tool ran (`agent_server.py:800-806`). The scratch repro (`scratchpad/repro/double_count.py`, real `_recorded` plus the end-of-turn logic) prints `unrelated_turns = 2` after one routed weather question during `researching`, and the chain is **gone** after the second. One "lights on and what's the weather" turn can reach 3 by itself. The unit tests exercise each site alone (`test_chain_wiring.py:97-108`), so they pass. `smart_home` is not wrapped, so home commands count once. The inconsistency itself shows this was not intended.

**D3 (medium). The `web_research` redirect throws away the model's resolved query.** When the user's words match `WHO_QUESTION`, the model's `web_research` call is rewritten to `web_answer` with `question.current`, the raw transcript (`agent_tools.py:966`). After "Who founded Nvidia?" → "Jensen Huang …", a follow-up "who's his wife?" is searched as "who's his wife?" even if the model wrote "Who is Jensen Huang's wife?". Fix: keep the model's `query` when it is a full question, and use the raw words only when it is empty.

**D4 (medium). The rules for restatements and acknowledgements reach Gemini only while a chain is active.** "No, I mean … corrects your last action" and "Got it / thanks start nothing" live in `system_agent_case_f.txt`. Production runs `--hedge 0 --router-url` (`scripts/stack/up.sh:143-144`). Hedged requests go out with the base prompt (`agent_server.py:560-566`; `hedge.py` never steers), so on ordinary turns Gemini never sees these rules. HANDOFF lists "restated requests loop on lookups" as open, and this is a likely contributor.

**D5 (medium). No date in the default front prompt** (`agent_server.py:870-873`). Date arithmetic and recency judgements ("is that still true", "how long ago") are answered from model priors. The model can call `get_current_time`, but nothing tells it to do so before reasoning.

**D6 (medium). History keeps what was said, not what was done.** The assistant side of each pair is the spoken text. A web answer's source, a tool's arguments and a delegate job id are gone on the next turn. Interrupted turns become `"(interrupted)"` even when a tool ran (`__main__.py:1234`). The server already keeps a reply-text → router-lines map (`agent_server.py:186, 489-491`). Annotating Gemini replies the same way would be cheap. The map is keyed by the exact reply text, and the text is rewritten after the fact (`strip_sources`, `dropped_sentences`, `agent_server.py:778-782`), so lookups can silently miss and fall back to `none|<100 chars>`. I have not verified how often that happens.

**D7 (medium). Split and cut utterances reach the model as fragments.** `complete_turn` returns at once without a turn model (`__main__.py:723-724`), and `talk.sh` does not pass `--ask-to-add`. Only barge-in joins exist. The 2026-11-12 research proposed a rule-based "ends on a conjunction or article" fallback; it was never built.

**D8 (low-medium). The location line is all or nothing** (`venue.py:105-114`). If the barangay lookup fails at boot, the prompt says nothing about where the user is. The weather tool defaults to home, but `web_answer` "here" questions and Case G get no place.

**D9 (low). The fast path ignores context.** "turn it up" or "turn it down" always means volume (`fast_path.py:107`), even right after "set the temperature to 24". "next" always skips the song.

**D10 (low). Token cost is unmeasured and doubled.** `--hedge 0` sends every Gemini turn twice. The system prompt is about 6.8 KB plus profile/venue/name lines (roughly 2k tokens with about 10 tool schemas), and Case F adds about 570 tokens on escalated or chain turns. Usage metrics are off, so the implicit-cache hit rate is unknown. This is not a bug, but every proposal below adds tokens and none can be priced today.

**D11 (low). A notice consumes a history slot as a fake user turn** (`__main__.py:675`). With a window of 4, one played notice evicts a real exchange.

**Fine as is (one line each):** the static system prompt with the volatile chain/protocol appended to the user tail is the right ordering. The doc-answer verbatim-quote check is a good model for web checks. Tavily's `unconfirmed()` hedge is sound. Chain blocks are budgeted.

## 4. Ranked priorities

Effort: S < 1 day, M 1–3 days, L > 3 days. Each "test" is a fixture that runs in CI, not a narrated run.

### P1. Grounded-answer enforcement, source return, structural post-check (the CEO case). S–M
- **What.** (a) Grounder prompt: replace "Do not search for settled facts" with OnIt's rule ("anything that can change needs a search: who holds a role, prices, versions, rankings, results"), and add the venue. (b) Keep `groundingChunks` (title, uri) and `groundingSupports` in `Grounded`, and **log the query** in the ledger. (c) A deterministic gate in `grounded()`: if the question is changeable (`one_fact_question` plus a role-noun list: CEO, president, chair, head, mayor, coach, minister, "current", "latest") and `searches == 0` or no support segment covers the answer, retry once with a forced-search instruction. If it still fails, fall through to Tavily (which always searches, then `unconfirmed()`). If that fails too, speak a hedge ("I couldn't confirm that with a source; my best guess is …"), which is rule 2. (d) Return a publisher phrase built by code from the first support's chunk title ("according to nvidia.com" → spoken "according to Nvidia's site"), consistent with the Case F "publisher in words" rule. Put the full titles and URLs in the turn log and in `NOTES.md`. (e) Never cache an unsearched answer to a changeable question.
- **Where.** `src/voice_agent/mcp_servers/gemini_grounding.py` (`SYSTEM`, `parse`, `Grounded`, `Ledger._write`), `tavily_search.py` (`grounded`, `remember`), `agent_tools.py` (role-noun classifier next to `one_fact_question`).
- **Payoff.** Closes rules 1, 2 and 5 for the most visible failure. Adds about 0.9 s to role questions only (zero-search 0.91 s median against one-search 1.8 s).
- **Test.** Save 10–12 real `generateContent` JSON responses once (Nvidia CEO, UP president, "last NBA champion", 3 settled facts, 1 response with zero searches) into `pipeline/tests/fixtures/grounding/`. Unit-test `parse`, the gate, retry selection and the spoken source phrase against them. Extend `scripts/search_eval.py` live mode with the assertion "changeable class: searches ≥ 1 and a source on 100 %" and report the latency delta.
- **Risk.** More searches spend the free 5k a month (the ledger cap exists). Chunk titles are sometimes bare domains, so keep the phrase generic. Over-triggering on "who is Rizal" adds latency only.
- **Depends on.** Nothing.

### P2. Fix the four cheap context defects (D2, D3, D4, D9). S
- **What.** D2: count unrelated turns in one place, the end of the turn, and delete the call in `_recorded`. D3: keep the model's `query`. D4: move the two rules ("No, I mean …", "Got it/thanks") into the base prompt, about 60 tokens, or into a regex-triggered tail suffix. D9: skip the fast path for "turn it up/down" when the previous assistant turn set temperature or brightness.
- **Where.** `agent_tools.py:963-983`, `agent_server.py:800-806`, `system_agent.txt`, `fast_path.py`/`agent_server.py:704-708`.
- **Payoff.** Chains survive side questions during research. Follow-up searches keep their referent. Restatements stop looping.
- **Test.** Turn `scratchpad/repro/double_count.py` into `tests/test_chain_wiring.py::test_one_turn_counts_once` (it fails today). Add a unit test that the redirect keeps `"Who is Jensen Huang's wife?"`. Add dry-run prompts (`config/eval/agent_prompts.jsonl`) for "No, I mean Taylor Swift" after a play request.
- **Risk.** Low. D4 adds tokens to every request.
- **Depends on.** Nothing.

### P3. Per-turn time and place tail, explicit "cannot" block. S
- **What.** Append one line to the last user message for every Gemini request, inside `steer()`, extended to the hedged path: `[Now: Saturday 14 November 2026, 7:05 PM Asia/Manila | Place: Cebu City (Barangay X)]`, about 30 tokens. It goes in the volatile tail, so the system prompt stays byte-stable. Fall back to the city when the barangay is unknown. Add a 4–6 line static "You cannot" block: no personal contact lookup, no purchases, cannot see a page's content except through its manifest, cannot reach the internet when offline, no memory of earlier sessions (until P6). Pass the venue to the grounder prompt.
- **Where.** `agent_server.py` (`steer`, and the hedged branch at `:560-566` needs a per-turn suffix too), `hedge.py` (accept a suffix), `venue.py`, `system_agent.txt`.
- **Payoff.** Rule 3 fully met; D5 and D8 closed.
- **Test.** Dry-run prompts with expected behaviour: "how many days until Christmas" (answer contains 41), "who is the mayor here" (`web_answer` query contains Cebu), "which barangay are we in" with the venue lookup stubbed to fail (says Cebu City). Unit test that `steer()` output ends with the Now line and the system string is identical across two calls.
- **Risk.** A per-turn timestamp in the user message does not affect caching, because the user tail is never cached anyway.
- **Depends on.** Nothing.

### P4. Make history carry what was done, and measure before widening. S
- **What.** (a) Turn on usage metrics (`Agent(usage=True)`) and log prompt, cached and completion tokens per turn. This is the prerequisite for pricing everything else. (b) Server side, annotate each assistant reply in the window from the server's own turn log (as `router_lines` does), e.g. `… [did: web_answer "CEO of Nvidia" → source nvidia.com]`, capped at 120 chars. Key the map on the *final* logged text so the lookup survives rewrites (D6). (c) Keep notices out of the user role: fold them into the previous assistant turn (D11). (d) Then A/B `--history 4` against 8 on a referring-back fixture before changing anything.
- **Where.** `agent.py`, `agent_server.py` (`history`, a new `annotate`), `src/doc_chat/__main__.py:675`.
- **Payoff.** Follow-ups like "where did you get that?" and "do it again" become answerable. This also produces the record format that P5 and P6 reuse.
- **Test.** A replayed 6-turn transcript fixture (from `out/sessions/*/turns.jsonl`) through `AgentServer.handle` with a recording agent. Assert the request messages contain the `[did: …]` annotation and no `(a background job finished)` user turn.
- **Risk.** Annotation text could be read aloud, but the payload and spoken filters exist. About 30 tokens per pair.
- **Depends on.** Nothing. P5 and P6 build on it.

### P5. Submit-time thread tracking (the owner's idea), with a fail-safe. M
- **First gate this on data.** Label about 100 turns from the 22 recorded sessions: new topic, continuation, or return to thread k. If returns beyond the 4-pair window are under about 5 % of turns, stop at P4. Sessions are long (median about 27 turns), but how often topics return is unknown.
- **Hook.** In `AgentServer.handle`, after `question.current` is set and before the routing branch (`agent_server.py:703-709`), call `threads.observe(said, messages)`. It is synchronous, pure Python, and must run in under 5 ms (budget 50 ms). Skip backchannels, and skip the fast path except for logging. At the end of the turn (`:789-799`) call `threads.close_turn(row)` with the reply, tools and sources (P4's annotation).
- **Record** (`<state_dir>/threads/<session>.json`, rewritten atomically, plus an append-only `threads.jsonl` event log):
  ```json
  {"session": "stack-1114-1609", "current": "t3",
   "threads": [{"id": "t3", "title": "nvidia ceo", "status": "open|dormant",
     "keywords": {"nvidia": 3, "ceo": 2, "huang": 2}, "entities": ["Nvidia", "Jensen Huang"],
     "first_turn": 4, "last_turn": 9, "last_at": 1763117000,
     "turns": [{"n": 9, "q": "...", "a": "... (<=200 chars)", "did": "web_answer → nvidia.com"}],
     "facts": [{"text": "Jensen Huang is CEO", "source": "nvidia.com", "at": 1763117000}],
     "chain": null}]}
  ```
  Caps: 8 threads per session, 6 turns per thread, 5 facts. Dormant means the thread's last turn is outside the 4-pair window.
- **Detection (lexical first, no model).** (1) If the turn has a back-reference (the `REFERENCE` regex in `call_policy.py:643`) or is short or a follow-up (`chain_state.is_followup`), it continues the current thread. (2) An explicit return marker ("back to", "going back to", "earlier you said", "about the X again", "what about the") boosts. (3) Otherwise score content words (the `tavily_search.STOP_WORDS` set) against each thread's keyword bag plus entities: BM25 or weighted Jaccard with recency decay. The best dormant thread wins if its score is ≥ τ and beats the runner-up by margin m. If nothing passes, it is a new thread. τ and m are fitted on the labelled set. Embeddings: **not at first**. MiniLM-class ONNX is about 10–30 ms on the laptop and plausibly 50–150 ms on the Pi 5 (unmeasured), which is outside the budget on the Pi. Add one only if the labelled set shows lexical recall on paraphrased returns under about 0.7.
- **Rehydration.** Only when the winner is a dormant thread. Append to the last user message (volatile tail, same place as the chain block): `[EARLIER TOPIC (may be relevant; ignore if not): <title> | Q: … A: … | facts: … (source)]`, at most 400 chars and the last 2 pairs. Because the router sees 1 pair, a rehydrated turn whose router lines are a lookup or `none|` goes to Gemini (same mechanism as `lookup_needs_context`). Facts carried from a thread are labelled with their age. A changeable fact older than about 10 minutes is shown as "earlier said", never as current, so P1 still searches.
- **Failing safe.** An ambiguous score (inside the margin), a home command, an active chain or any exception: no rehydration, and the turn is byte-identical to today. The tracker only ever *adds* a block and never removes history. Behind a `--threads` flag that defaults off until the fixture passes. Every decision is logged on the turn row (`thread`, `score`, `rehydrated`) for audit.
- **Relation to `chain_state`.** Keep them separate. The chain is a phase machine for one task; threads are topic memory. A thread stores a chain snapshot so "back to the page" can re-offer the artifact.
- **Test.** The labelled set becomes `pipeline/tests/fixtures/threads/*.jsonl`. Assert precision ≥ 0.9 on "rehydrate" decisions and report recall; assert the p99 of `observe` stays under 5 ms on 89-turn sessions. One replayed scenario: CEO question, 5 unrelated turns, then "and how old is he?" must carry Jensen Huang into the request.
- **Risk.** A wrong rehydration steers the answer to the wrong topic, which is why the precision floor and margin exist. Adds a state file per session.
- **Depends on.** P4 (turn records with "did" and sources) and the labelled set.

### P6. OnIt-style episodic memory across sessions, adapted to a voice latency budget. M
- **What.** `episodes.jsonl` in the state dir, one line per finished turn worth keeping (P1-verified web answers, delegate results, user-stated preferences, corrections): `ts, session, owner, model, question, answer, did, sources, verdict (grounded|unverified|corrected)`. Load it into memory at boot. Use BM25 (`rank_bm25` is already a dependency through `docs.py`'s `BM25Retriever`); on the order of 10k rows takes a few ms. Filter by owner and by model as OnIt does, and rank verified above unverified within a similarity band. Hard cap 300 tokens, untrusted-data marker, volatile tail.
- **Recall only on trigger, not every turn.** (a) Back-reference cues to past sessions ("last time", "you told me", "yesterday", "remember when"). (b) "What did you find about X". (c) P5 finds no in-session thread but an episode scores above a high threshold. An explicit "remember that …" is written by code from a regex. No new model tool (see P8).
- **Rule.** An episode never answers a changeable fact. It gives context ("you asked this on Tuesday; it was Jensen Huang, per nvidia.com"), and P1 still searches.
- **Where.** A new `pipeline/src/voice_pipeline/episodes.py`; hooks at the same two points as P5.
- **Payoff.** Rule 4 met in OnIt's shape, at a few ms per turn.
- **Test.** Fixture episodes (with two owners and two models) plus queries. Assert the filters, ranking, budget truncation (whole episodes only) and the "changeable fact still triggers a search" path. Latency assertion at 10k rows.
- **Risk.** Privacy: the owner's queries are persisted, so keep it in the state dir, out of git and out of promote. Stale facts are handled by the rule above.
- **Depends on.** P1 (verdicts) and P4 (records). Can share P5's tokenizer.

### P7. Volatile-last ordering, verified. S (measurement)
Already mostly right. After P3 and P4 add tail content, confirm with P4's usage logging that cached tokens on turn n+1 cover at least system plus tools, and keep the system string free of per-turn bytes. A test asserts that the `build_llm` system string is identical across two simulated turns. No redesign.

### P8. Model-callable notes / `context_status`: do **not** build for the front agent.
A front-agent turn holds about 2–3k tokens, makes 1–3 calls, and never compacts, so `context_status` has nothing to report. Each extra tool costs schema tokens on every request (doubled by hedging) and selection accuracy; the 10-tool idle cap was a deliberate decision. The useful slice ("remember that …") is a regex handled by code in P6. The delegate (long runs) is where OnIt's notes tools would pay, and it already has `NOTES.md`.

### What I would NOT do, and why
- **LLM summarization or compaction of history.** At 4 pairs (about 300 tokens) there is nothing to compact, and a summary call adds 0.5–1 s or a background job with staleness. P4 and P5 recover the lost turns deterministically.
- **A vector DB or archival memory.** BM25 over a jsonl covers a single-owner voice log. The 2026-11-12 research already rejected it, and nothing here changes that.
- **An LLM thread classifier per turn.** It would blow the 50 ms budget. Lexical first, embedding only if the fixture demands it.
- **OnIt's LLM `verify_answer` on every reply.** Our answers are under 80 chars, where OnIt itself skips. The structural gate in P1 (search happened, support covers the answer, names present) does the work at about 0 ms. Keep an async LLM check only for `web_research` readbacks, if ever.
- **Raising `--history` blindly.** Measure first (P4d).
- **Merging threads into `chain_state`.** Different lifetimes and semantics; merging would put the task protocol at risk.

## 5. Papers and open questions

I have **not** read the four papers, and I cannot vouch for the two newest IDs (2601.03785 Membox, 2609.33226 Threader). 2609.* is after my knowledge, so treat any summary of it as unverified. What to extract from each:
- **RMM (2503.08026, reflective memory management).** As I recall, prospective reflection (summarising dialogue into topic-keyed memories at session end) and retrospective reflection (reranking retrieved memories using whether the answer actually cited them). Take: topic-keyed granularity for P5/P6 records and the "was it used" signal as a cheap outcome label. Skip: the RL reranker.
- **Membox.** Read for the memory unit (turn, topic box or session) and the write policy (what is not stored). Check whether it reports latency.
- **Threader.** The closest to the owner's idea. Read for how threads are detected (per-turn classifier vs clustering), the reported accuracy, the per-turn cost, and the failure handling when it misassigns. If detection needs an LLM call per turn, it does not fit a 50 ms budget as is.
- **HiMem.** Read for the hierarchy (raw turns → topic → profile) and when higher levels are rebuilt. Our hierarchy would be the window (4 pairs), threads (session), episodes and profile (cross-session).
For each, record its benchmark (LoCoMo, LongMemEval, MSC?) and whether gains hold on short, spoken, task-oriented turns. Most are evaluated on long chit-chat, which is not our distribution.

Verify empirically before building:
1. Of the 208 zero-search grounded answers, how many were changeable questions and how many were wrong? This needs the query column (P1b). Then re-measure the zero-search rate after the prompt change.
2. The latency and free-tier cost of forcing a search on role questions (`search_eval.py` live, about 20 calls).
3. How often a topic returns after falling out of the 4-pair window, from labelled `out/sessions` (gates P5).
4. Lexical detector precision and recall on that labelled set; embedding latency on the Pi 5 if lexical recall is under about 0.7.
5. Whether flash-lite uses a rehydrated `[EARLIER TOPIC]` block correctly and ignores it when irrelevant (a replayed fixture, about $0.02 a trial).
6. History 4 against 8: resolution accuracy on referring-back fixtures, plus prompt tokens and first-token latency from P4a.
7. The implicit cache hit rate on gemini-3.1-flash-lite at our prompt size (a minimum-token threshold may apply; unknown).
8. How often `router_lines` / annotation lookups miss because the reply text was rewritten (D6): count `none|` fallbacks in logs after P4.

Side effect to note: running the D2 repro made `uv` install the pipeline venv (git-ignored). The working tree is still clean.
