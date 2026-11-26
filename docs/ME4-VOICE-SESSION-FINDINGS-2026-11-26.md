# ME4 spoken session findings (2026-11-26, Pi 4, 48-file corpus)

Why this note exists: the owner's first spoken run gave "very bad results on the x questions", and standard statements such as "Good evening" got a document call. Evidence: the Pi agent server's own log of that session (60 turns, `pipeline/out/agent_server/20261126-173512.jsonl` on the Pi), then a replay of 12 utterances through the same server on the latest code. Nothing in the repo was changed. The three causes below are open.

## What was running
- The Pi agent was at `8dbc4e3` (fix 234). Fixes 235 (non-questions skip documents), 236 and 239 were **not deployed**. That alone explains the pleasantry document calls in the session. The Pi now runs `8cb8f94`.
- Stack: HARNESS=pi, `--docs-first --hedge -1`, router on the laptop, 48-file index (13,898 chunks). Speech recognition is Parakeet with the 325-name `entities.json`.

## Session in numbers (60 turns, routes from the jsonl)
`docs_miss` 16, `chain` 16, `docs` 12, `routed` 12, `backchannel` 2, `fast` 1, `docs_general` 1.

| what the owner said | route | result |
|---|---|---|
| "Hello, good evening." / "Good evening." | docs_miss | "Your documents don't cover that. Good evening." or the offline sentence |
| "Sorry, I mean T T S." / "Can you try again?" | docs_miss | a document call and prefix on a repair phrase |
| turn 45 onward, every turn to the end ("Who is the current Nobel Peace price Laureate?" ... "Got it. Goodbye.") | **chain** (16 of 16) | the document path is never reached |

## Cause 1: an open research chain captures every later turn (the x questions)
- `agent_server.py:1177-1184`: when `chain.active()` is true, the turn goes straight to Gemini with route `chain`. This branch sits **before** the docs-first block (`agent_server.py:1187`), so no document check can run while a chain is open.
- `chain_state.py:26-37,243-253`: a chain expires after 600 s or after 3 unrelated turns. `web_answer` is in `CHAIN_TOOLS`, so every web answer counts as chain work, not as an unrelated turn, and it also refreshes `updated_at`. A run of web questions keeps the chain alive indefinitely.
- The session shows it: from "Who is the current Nobel Peace price Laureate?" to the last turn, 16 consecutive turns were `chain`, including the `x` questions that need a document (census file, briefing notes, ATSDR module). They were answered by Gemini and the web with only the filler "Checking your census file ...".
- Wrong answers that follow: "The UK prime minister in my briefing notes, is that still right?" was answered from the web only (the briefing note says Keir Starmer; it was never read). "The ATSDR module mentions a law from 2010. Who signed it?" then "Who signed it?" gave "President Jimmy Carter signed the Comprehensive Environmental ... act".
- Replay after a restart showed the same thing: the leftover chain turned "Hello, good evening." into a Miss Universe answer ("Good evening. Jennifer Barreto of Puerto Rico won Miss Universe 2026 ...").
- Category: context carry-over (stale task state), not retrieval.

## Cause 2: the new non-question gate drops real questions with a spoken lead-in (235 over-blocks)
`doc_route.py:602-617` (`question_like`) only accepts an utterance that **starts** with a question word, after an optional "so / and / ok / hey / please / can you tell me ...". The agent server then skips documents when `question_like` is false and `document_cue` finds no file name (`agent_server.py:1297`).
I ran the current classifier over all 60 session utterances. These real questions would skip the documents (`ql=0 cue=0`):
- "Got it, what's the office Wi-Fi password?", "Understood, who is Japan's prime minister?", "Sorry, I mean who won the latest mayor race in LA?", "Alright, ..." (spoken lead-ins "got it / understood / sorry / alright").
- "On which day do NYU Abu Dhabi classes follow a Monday schedule?" ("On which" is not in the start list).
- "Broadly speaking, what are all the documents about?" (the general lane is never reached).
- "The At SDR module mentions a law from 2010. Who signed it?" (not phrased as a question word).
- "This was California's commercial electricity price in August 2026." (garbled start).
Pleasantries are correctly skipped ("Hello, good evening.", "Good evening.", "Say OK.", "Sorry.").

## Cause 3: speech recognition errors on proper nouns and numbers
Heard text from the session: "T P S" for TTS, "Wi Pi" for Wi-Fi, "per diamond" and "per dime" for per diem, "At SDR" and "Tentitan" for ATSDR and 2010, "marriage race" for mayor race, "price Laureate" for prize, "class sample" for QCAS sample, "Community x7" noise. The vocabulary file repaired "TTS" on the repeat, but not these.

## Replay on the latest code (8cb8f94), one utterance each
| utterance | route | answer, short |
|---|---|---|
| "Got it, what's the office Wi-Fi password?" | routed | search_docs tool, "The documents don't say what the office Wi-Fi password is." |
| "Understood, who is Japan's prime minister?" | routed | web, Sanae Takaichi |
| "Sorry, I mean, the ATSDR module mentions a law from 2010. Who signed it?" | routed | search_docs, long answer; the signer was not stated in the part logged |
| "On which day do NYU Abu Dhabi classes follow a Monday schedule?" | routed | "As per liberal_studies_academic_calendar.html: ... July 7, 2027" (**wrong**; the golden answer is October 14, 2026) |
| "Broadly speaking, what are all the documents about?" | routed | docs_catalog tool, a summary (works, but via the router path) |
| "The UK prime minister in my briefing notes, is that still right?" | routed | "As per Wikipedia and WWW, the current Prime Minister ..." (briefing note ignored) |
| "Thanks.", "Okay.", "Got it. Goodbye." | chain or routed | no document call once the chain is idle |
Several of these now skip the checked document path (Cause 2) and the NYU Abu Dhabi date answer shows the cost: an unchecked answer from the wrong calendar year.

## Proposed fixes (not made; for the orchestrator or Qwen)
1. **Chain capture:** move the docs-first/document-cue check before the chain branch, or release the chain when the utterance has a document cue, or stop counting `web_answer` as chain work so unrelated turns can expire it. Also reset chain state when the agent service starts. Test: an open research chain, then "The UK prime minister in my briefing notes, is that still right?" must reach `ask_documents` (route `docs` or `docs_miss`), never `chain`.
2. **Question gate:** strip leading fillers before `question_like` ("got it", "understood", "alright", "sorry", "sorry i mean", "right", "so", "well", "now") and add "on which", "in which", "during", "for which"; treat a sentence containing a wh-word or auxiliary after a comma as a question. Test: the 60 session utterances, with the expected skip set equal to the pleasantries only.
3. **Vocabulary:** add "Wi-Fi", "per diem", "ATSDR", "QCAS", "mayor race" variants to the entity list or the ASR hotword list; re-check on the session recording.
4. **Deploy discipline:** the Pi must run the same commit as the tests. Record `~/me4-ingest/.commit` in every report. `git archive HEAD` over ssh is the working method.

## Not verified
- The exact turn that opened the chain (probably the Miss Universe lookup): the jsonl does not log chain phase per turn.
- Whether an `x` question that starts a document answer reaches the checked path on an idle chain after fix 2; the replay above used an idle chain after the first turn.
- Model variance: each replay utterance was run once.
