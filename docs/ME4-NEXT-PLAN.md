# ME4 next work plan (rev 3, Opus critic, 2026-11-20): act-then-critique, multi-source answers, zip input, cold start

Author: external critic (Opus), 2026-11-20, read-only on `local/agent-assessment` @ `b7edff1`.

Inputs:
- `docs/ME4-SPEC.md`, `critic-opus.md`, research reports A-E, the code at HEAD.
- OnIt's act-then-critique: `onit/src/model/serving/verify.py`, `default.yaml` (verify_*), `onit.py` (background checks), `SELF_IMPROVEMENT.md` thesis 4.

Rev 2 changes:
- Act-then-critique added to the product (WP8, rewritten) and to the development process (§1.0).
- After an adversarial pass, the ordering and scope changed. §6 logs every finding.

Rev 3: a cold recheck with code verification, logged in §7.

Tags: **VERIFIED** = read in code at HEAD or probed; **HYPOTHESIS** = not checked.

## 0. What decides the day

The checker unzips a folder and asks questions that may need one document, several documents, or documents plus the web. In order, the things that lose points:

1. **Document-answerable questions that get answered without the documents.**
   - A question without "my documents" goes through the implicit gate (coverage ≥ 0.55, `doc_route.py:221-230`).
   - If it fails, the turn falls to the router or Gemini. Routed web lookups do race the documents (`agent_server.py:506-521`, `DOCUMENT_LOOKUPS`), but direct router replies (`none|reply`) never consult the documents, and Gemini escalations consult them only if Gemini chooses `search_docs`. VERIFIED.
   - The prompt for document questions says "Call search_docs once ... never web_answer" (`pipeline/config/eval/system_agent.txt:20`). That blocks documents+web. VERIFIED.
2. **Multi-source answers.** `DocAnswer` holds one quote and one passage.
   - `search_docs` is called with no `top_k`, so k=4, with a cap of 2 per file (`doc_route.py:267`, `docindex.py:44-45`).
   - `names_numbers_ok` checks against *all* passages joined (`doc_answer.py:375-376`), so a claim can borrow a number from another file.
   - The doc/web race has exactly one winner (`agent_server.py:439-449`).
   - All VERIFIED.
3. **The zip itself.** Nothing extracts a `.zip`; a dropped `.zip` becomes an "unsupported extension" failure (`docindex.py:2225-2240`). VERIFIED.
4. **Tables.** CSV *and* xlsx chunks lose their row label.
   - CSV probe: 7,831 chunks.
   - xlsx probe: a 41×61 sheet makes 123 chunks, and 75 of them lack the row label. VERIFIED.
   - Nothing aggregates (sums, maxima).
5. **Cold start.**
   - The first refresh publishes only at the end (`docindex.py:2376-2384`).
   - `Retriever.build` and the cache prune run under `_lock`, which blocks `search()` (Pi: 4.7 s on the GDP CSV, per the fresh-file table).
   - VERIFIED by reading.
6. **Budget.** The doc answer has 2.5 s for search plus one unhedged Gemini call (`doc_answer.py:20,412-432`). Warm Gemini is p50 0.98 s / p90 1.31 s with 3/60 over 2 s (`docs/LATENCY.md:23`). Longer multi-claim outputs will push more answers past the budget. HYPOTHESIS until WP0 measures it.

## 1. Work packages

Who does it:
- **S-dev**: Sonnet `developer`, working from a ticket.
- **H/agy**: Haiku or agy, spoon-fed (per memory: two failures in a row and Opus implements).
- **Opus**: design and critique.

### 1.0 Act-then-critique in the development process (applies to every package)

Each package is accepted only after **act → critique against evidence → accept / fix**:

- **Act.** The implementer lands the diff, the tests, and a measurement file (the WP0 or metrics JSON named in the package).
- **Critique.** An independent critic, never the author, does three things:
  - re-runs the named tests and the measurement command;
  - compares the numbers with the package's acceptance thresholds and with the previous baseline (the paired 112-question set, WP0 timings);
  - reads the diff for the risks named in the package.

  Narrated results do not count; only command output does (agents.md principle 10). A finding needs evidence: a failing test, a number outside its threshold, or a diff line. Unbacked doubts are advisory, the same fail-open rule as the product (WP8).
- **Who critiques:**
  - **Opus**: WP2 (locking), WP3 (answer contract), WP8 (product critic), WP1 (alongside `security-auditor`).
  - **Sonnet tech-lead review**: WP0, WP5, WP7, WP9, WP11.
  - Output from Haiku or agy (WP6, WP7 readers): Sonnet critique plus the numbers re-run by the orchestrator.
- **Retry cap:** 2 fix rounds, then escalate to the owner.

### WP0. Rehearsal harness and frozen question set (the producer of all measurement)

**Design: new `scripts/me4_rehearsal.py`.**
1. Extract the zip. Use WP1's `zipingest.extract_zip` once it lands; before that, plain `zipfile` into a temporary folder. A second commit is required: the harness must switch to the real extractor.
2. Start the real docs server over stdio with the `mcp` client already in the root venv. This is the first MCP round trip ever measured.
3. Poll `docs_status` once a second. Record `first_searchable_s` and `ready_s`.
4. For each question:
   - call `search_docs`, then `doc_answer.answer` with `bench_fakes.fake_doc_request` (offline) or `gemini_request` (live, approval-gated, disk cache keyed by sha256 of model+prompt);
   - score it: gold file present in the passages; `answer_regex` matches; `judge` accepts; abstain phrase on unanswerable items;
   - record timings per stage, and the WP8 verdict labels once WP8 lands.

**Corpora:**
- **mock-day zip:** about 25 files from `ipo-2026` plus 5 hand-made office/csv/xlsx files.
- **nasty zip:** cp437 names, `__MACOSX`, a nested zip, a 30 MB PDF, a scanned PDF, xls, rtf.

**Questions:** 40, written by the owner or Opus, not by the implementer. Mix: 12 single, 8 multi, 5 table lookup, 3 aggregate, 4 doc+web, 8 unanswerable. Freeze them before WP3.

- **Accept:** offline run under 5 min on x86; warm rerun under 30 s; JSON report.
- **Effort:** M. **Risk:** low. **Who:** S-dev. **Critique:** Sonnet.

### WP1. Zip ingestion (R2)

**Design: new `src/voice_agent/rag/zipingest.py`, `extract_zip(zip, corpus) -> manifest`.**
- Use the `zipguard` *checks* (ratio, member count, path) but give archives **their own limits**. Zipguard's limits are sized for one OOXML file: `MAX_ZIP_TOTAL_BYTES` 64 MB, members 20 MB (`zipguard.py:13-16`). The ipo-2026 corpus alone is 71 MB, so those limits would refuse a realistic checker zip. Proposed archive limits: total 1 GB, member 200 MB, 5,000 members, ratio 100. Per-file `MAX_FILE_BYTES` still applies at index time (§3 Q5).
- Refuse a member unless `resolve().is_relative_to(dest)`.
- Reject symlinks (external_attr), encrypted members and nested archives.
- Skip `__MACOSX` and `._*`.
- Name decoding: UTF-8 flag, else strict UTF-8, else cp437.
- Extract to `<corpus>/<stem>.tmp-<pid>/`, then `os.rename` to `<corpus>/<stem>/`. The rename is atomic, so the indexer never sees half-written files.
- Called from `docs.py` `_Indexer._run` before `refresh()`. Extracted zip hashes go in `.rag_cache/zips.json`.
- `--corpus x.zip` in `main()`.
- Refused or skipped members appear in `docs_status.failures` as `zip: <why>`.

Report C is wrong on two counts:
- The readers take bytes, not paths (`convert.py:732-753`, `docindex.py:2258`).
- A staging folder under `.rag_cache` would be invisible, because the walker skips dot-directories (`docindex.py:1564-1565`).

- **Accept:** malicious fixtures (zip-slip, absolute path, symlink, ratio bomb, 10k members, encrypted, cp437/cp932 names, case-duplicate names, nested zip) are all refused or reported, and nothing is written outside the destination. Every mock-day member is `ok` or carries a named failure.
- **Effort:** M (1.5-2 d). **Risk:** security. **Who:** S-dev. **Critique:** Opus + `security-auditor`.

### WP2. Cold start and the pipeline side of the indexer (ticket 23, extended)

**Design:**
- **(a) Incremental publish.** In `DocIndex.refresh` (`docindex.py:2121-2393`), swap a snapshot after each file, or every 2 s while parsing.
- **(b) Build off the search lock.** The FTS5 factory shares one instance (`backends.py:73-76`), so:
  - make `Fts5Retriever.build` sync only the dirty files;
  - derive keys from `chunk.chunk_sha256` plus the header, not `_chunk_key` over all chunks (`fts5.py:275`);
  - keep `rank()` usable during the sync;
  - move `canonical_cache.prune` out of `_lock`.
- **(c) Timeouts.** `asyncio.wait_for(…, 0.5)` on every `docs_status` call: `doc_route.py:254-256,286-292` and `agent_server.py:915`. On timeout, fall back to `cached_files()`.
- **(d) Honest "still reading".** `parse_docs_status` currently drops `indexing`/`ready`/`parsing` (`doc_route.py:155-189`), so extend it to return them. If the status says `indexing` and nothing was found, say "I'm still reading N files". Never say "the documents don't say" before `ready`.
- **(e)** `_settle()` skips the nudge when the last refresh finished under 1 s ago.

- **Accept:**
  - A term in file 3 is searchable while file 1 (the IRS PDF) is still parsing.
  - With the GDP CSV, `search_docs` stays under 200 ms on x86 throughout.
  - A fake `docs_status` that sleeps 5 s still leaves the turn under 3 s.
  - The 112-question paired set has zero lost rows.
- **Effort:** M-L (3-4 d; (b) is real concurrency work). **Who:** S-dev. **Critique:** Opus.

### WP3. Documents-first answers with multiple quotes: the ACT half (R5, R6; absorbs old WP4)

**Design:**
- **(a) Docs-first without breaking ME3.** In checking mode (`VOICE_AGENT_DOCS_FIRST=1`), every non-fast turn takes the implicit path with θ=0.
  - The doc attempt **races** `normal_turn()`: start both, and the document answer wins only if it is accepted before `claim()`.
  - On `none` it falls through silently, never with `NOT_FOUND_OFFER`. Otherwise "what's the weather" would get "the documents don't say" (VERIFIED `doc_route.py:341`).
  - Explicit document questions keep `handle_docs_turn`.
  - The race is what makes docs-first affordable, which is why the old WP4 merges here.
  - Plumbing risk (VERIFIED): `routed_turn` takes no `claim` parameter (`agent_server.py:599-601`). Only `run_routed` has one (`:462`), and `gemini()`/`agent.ask` has no claim hook outside `HedgedAgent`. The race needs a claim passed through `normal_turn` and a cancel for the Gemini leg. Effort is counted in WP3.
- **(b) Retrieval.**
  - `top_k=8` on the doc route.
  - Strip the document-reference phrase (reuse `_EXPLICIT_DOC`).
  - Split "A and B / compare A with B / A vs B" by rule into at most 2 local sub-queries, then merge (still 2 per file). No extra LLM call.
- **(c) One Gemini call, line schema.**
  - Format: `CLAIM n: … | P<k> | QUOTE: …` (at most 3 claims), an optional `MISSING: …`, or `NONE`. `max_output_tokens` about 160.
  - Per claim (the pre-act gate, which **fails closed** as it does today):
    - `quote_ok` against the **cited** passage;
    - `names_numbers_ok` against the **cited** passage (both helpers take plain strings, so per-claim use needs no rewrite, `doc_answer.py:241,248`; the new part is the line parser that replaces `parse`/`_REPLY`);
    - quote length within limits;
    - no directive.
  - Failing claims are dropped. If none survive, abstain.
  - Hedge the doc request (two requests, first wins) if the owner approves (§3 Q8).
- **(d) Voice rendering.**
  - Speak claims as soon as they pass the gate, documents first.
  - Speak at most 2 claims: "As per <file basename>: <answer>." (§3 Q9 decides whether quotes are spoken). Use the basename because nested zip folders make relative paths long to say; the text shows the full path.
  - The text and trace carry the full citation and quote.
  - Then, if there is a `MISSING:` line, run `web_answer(MISSING)` under a 2 s cap and say "From the web: … (source)". Web text is never merged into a document sentence.
  - Change the prompt line at `system_agent.txt:20` to "search_docs first; web_answer only for what the passages lack, and say which part is from the web".
- **(e) Conflicts.** Two surviving claims with ≥2 shared content words and different `_NUMBER` sets: speak both, each with its file.

Corrections to report A:
- A two-call decomposition does not fit the budget.
- A's support test ("claim words contained in the quote") rejects correct paraphrases.
- A missed the exclusive race, the prompt rule, and the joined-passages borrowing.

- **Accept:**
  - About 30 scripted-reply fixtures pass: fabricated quote, cross-file citation, two-file answer, conflict, `MISSING`, zero claims.
  - A weather question in docs-first mode gets the ME3 answer with no document sentence.
  - Benchmark: new multi-doc cases; `abstention` 16/0/3 unchanged.
  - Retrieval A/B, `top_k` 4 vs 8 on the frozen set: gold files present ≥ baseline.
- **Effort:** L (6-8 d). **Risk:** answer-contract and latency regressions. **Who:** Opus designs schema, prompt, judge and fixture specification; S-dev wires it up. **Critique:** Opus.

### WP5. Table chunking, CSV and xlsx (F6, report D rank 1)

**Design:**
- A new `table_specs(headers, rows)` feeds both `parse_csv_text` users (`docindex.py:317-340`) and the xlsx direct output (`readers._render_xlsx_sheet` and its chunking).
- Each chunk body is `<row label>: <col>: <val>; …`, grouped to fit `CHUNK_SIZE`. Narrow tables (fewer than 12 columns): 5-10 rows per chunk with the header repeated.
- Truncation becomes a `docs_status` note.
- **Bump `LEGACY_PARSE_VERSION`** (VERIFIED: the parsed CSV text itself is what gets cached, `docindex.py:1934-1941`) and `canonical.POSTPROC_VERSION`.

- **Accept:**
  - GDP CSV under 800 chunks, every chunk containing `Title: \d{4}`.
  - The xlsx probe sheet: every chunk carries its row label.
  - A 3,000-row CSV reports truncation.
  - Paired 112-question set: no loss in buckets other than csv/xlsx.
- **Measure:** `python -m voice_agent.rag.metrics --per-query`, both `--match` modes; `rag_paired_compare.py --metric hit4`.
- **Effort:** M+ (3 d). **Risk:** the repeated header inflates scores. **Who:** S-dev. **Critique:** Sonnet.

### WP6. Tokenizer folding (F4)

- **Design:** NFKD, drop `Mn`, map ligatures, then lowercase, in `corpus.tokenize`, `doc_answer.norm` and `_content_words`.
- **Add accented and ligature words to `_FINGERPRINT_PROBE`** (`fts5_store.py:30-32`). The fingerprint hashes `tokenize()` over an all-ASCII probe (`:71-76`), so folding alone would *not* change it, and existing on-disk FTS5 indexes would keep the old tokens (VERIFIED).
- Report D's porter flag is not an end-to-end test: search coverage, the implicit gate and `doc_answer` stay ASCII.
- **Accept:**
  - 10 accented golden rows: 0/10 → at least 8/10.
  - Paired 112 rows: 0 lost.
  - `beir_calibrate.py` SciFact/FiQA: no more than 0.005 lower. This needs the BEIR data prepared once by `beir_prepare.py`, a free download; the run itself is offline.
- **Effort:** S. **Who:** H/agy. **Critique:** Sonnet plus orchestrator re-run.

### WP7. Additional formats and honest failures (R3)

- **Formats** (in `readers.py` plus the tables at `docindex.py:49-85`):
  - tsv: through the table path from WP5;
  - json/jsonl and xml: guarded;
  - log/ini/toml/yaml: as plain text;
  - rtf: `striprtf`;
  - ODF: via zip and `content.xml`;
  - xls: `xlrd` (§3 Q4).
- **Scanned PDF:** fewer than 20 characters per page on more than half the pages → `unreadable: scanned PDF (no OCR)`. A partly scanned PDF is indexed and reports `ok (N pages without text)`; today those pages are silently skipped (`docindex.py:1125,1176`).
- **Order:** after WP5. **Effort:** M. **Who:** H/agy, one format each. **Critique:** Sonnet + `security-auditor`.

### WP8. Act-then-critique for document and document+web answers (the CRITIQUE half; absorbs old WP8, with answerability (D2) reduced to labels)

OnIt's split, mapped onto our pipeline. The **pre-act gate** (WP3c) decides *whether* we speak; it fails closed. The **critique** checks *what we spoke*; it fails open.

**Stage 1a: deterministic clearance (free, about 5 ms, runs before the turn's `done`).**
- A claim is **cleared** when every answer-bearing token is inside its *own* quote: numbers (`_NUMBER`), capitalised names not in the question, and dates.
- A web claim is cleared when it carries a source (`agent_tools.turn_source_sentence`).
- If every claim is cleared, no call is made. This is OnIt's `covered_by_trusted_sources`, with "trusted" meaning the user's files.
- It replaces the old WP8 idea of *rejecting* such answers. That would have caused false abstentions. Here an uncleared claim is only *checked*.

**Where the critique runs (VERIFIED constraint).**
- `handle()` passes text through `SentenceFilter`, which holds the trailing fragment until `sentences.finish()`. That call happens only after the turn's work returns (`agent_server.py:808-811,956-960`).
- Probe: a `format_citation` answer emits "As per … pesos. " immediately, and the closing `"quote"` is held until `finish()`.
- So a critique run *inside* the route would delay the quote by up to its full deadline.
- Design: the route returns its answer together with a `critique` coroutine. `handle()` flushes `payloads`/`sources`/`sentences` first, then awaits the critique, then sends any correction, then `{"done"}`.
- `handle()` holds `self.lock` for the whole turn (`agent_server.py:824`). The critique must therefore be cancelled as soon as another request arrives: `handle()` bumps a `self.waiting` counter before taking the lock, and the critique polls it. A cancelled critique is labelled `preempted`.

**Stage 1a+: local cross-file conflict check (free).**
- The `search_docs` MCP call with the claim text (the pipeline has no in-process index), with a 300 ms cap.
- If a passage from another file gives a different `_NUMBER` set for the same subject (≥2 shared content words), append "Another file, X, says Y."
- This was in stage 2 in the earlier draft. It moved here because the new cancel-on-next-turn rule would usually kill a conflict notice that waited for idle.

**Stage 1b: one small verdict call, only for uncleared claims.**
- **When:** after the flush, so all of the answer audio is already queued at the client. It runs in parallel with TTS playback.
- **Input:** the question, the cited passages only (at most 1,200 characters each), and the spoken claims.
- **Output:** JSON `{"issues":[{"claim","problem":"contradicted|unsupported|off_question","correction"}]}`. Thinking set to MINIMAL; `max_tokens` 120.
- **Deadline:** 1.5 s.
- **Acting on it:** only `contradicted` is spoken. `unsupported` is dropped, as in OnIt without lookups. A third type, `off_question` (the quote is real but does not answer the question, which is D2), is **label-only by default**: a wrong retraction would make a correct answer worse. Speaking it is §3 Q10.
- **Correction:** spoken as one appended sentence before `done`: "Correction: the file says <correction>." It is capped at about 20 words and is not a rewrite.
- **Cost:** about $0.00008 per call (`LATENCY.md:24`). If about half of document answers need it, that is roughly $0.004 per 100 turns.
- **Fails open:** a timeout, a parse failure, an empty correction, or a correction whose numbers appear in no cited passage all leave the answer unchanged.

**Stage 2: background, behind the answer.**
- Web parts reuse the existing `check_in_background` → `web_verify` (`agent_server.py:662-706`, VERIFIED). OnIt's design already exists here, for web answers only.
- **New: cancel on the next user turn.** Today `self.checks` tasks run for up to 20 s and their notices play even after the user has moved on (VERIFIED: tasks are only discarded when done). Also drop unacked correction notices that belong to an older turn.

**Labels.**
- Every verdict (cleared / clean / contradicted / timeout / unreadable) goes into the turn row and `checks/<run>.jsonl`.
- These become a free outcome signal for WP0/WP9 (OnIt thesis 4: 820 clean, 70 issues, 120 unchecked out of 1,010 were labelled this way).

**Latency budget for a spoken document turn:**
- search ≤ 0.3 s (Pi);
- Gemini answer ≈ 1.0-1.3 s warm (more with multiple claims: HYPOTHESIS, WP0 measures it);
- gate ≈ 5 ms → first audio at about 1.5-2 s.
- The critique adds 0 to first audio, and also 0 to the rest of the answer once the flush comes first (WP8 test). It adds at most 0.3 + 1.5 s to `done`. It sits outside `DOC_ANSWER_TIMEOUT_S` because `doc_route` wraps only the answer, not the critique, in `wait_for`.
- The client speaks sentences as they stream (VERIFIED for `doc_chat` via `speak_sentences`; `pi_speaker` HYPOTHESIS, verify in WP11).
- Report B's extra "does the quote answer?" call is not placed in front of speech.

**Accept: offline tests with scripted verdicts (`bench_fakes`) and injected faults:**
- digit drift (answer 1,599, quote 1,509) → verdict contradicted → correction spoken;
- a fully cleared answer → zero calls (counted by the fake);
- verdict timeout, garbage, or `unsupported` only → answer unchanged;
- a correction whose figure is in no passage → ignored;
- a stale background notice after a new turn → cancelled;
- two-file conflict → appended sentence;
- the answer text frame is sent, and the held quote flushed, *before* the fake verdict starts (the fake records order);
- a new request arriving during the verdict → critique cancelled within 50 ms, labelled `preempted`.

**What WP8 does NOT fix: D2.** The clearance rule passes a real quote whose figures are all inside it, even if it answers a different question. Only `off_question` labels measure this. The D2 xfail stays an xfail. Its rate is reported by WP9 live, and §3 Q10 decides whether it is spoken.

**Measure:** per-turn critique labels in WP0 (offline) and WP9 (live).

- **Effort:** M (3-4 d). **Order:** after WP3 (needs per-claim structure). **Who:** Opus designs prompt and rules; S-dev implements. **Critique:** Opus.

### WP9. Live evaluation (R9), approval-gated

- WP0 in live mode.
- At most 3 calls per question (answer + verdict + web); 5 calls if Q8 hedging is on. 40 × 5 × $0.00008 ≈ $0.016.
- Response cache; deterministic scoring; Wilson intervals; McNemar for any A/B.
- Project the cost from the first 5 calls, and stop at a $0.05 cap.
- **Effort:** S. **Owner approval required.**

### WP10. sqlite table tool (optional, after WP5; only if WP9 shows aggregation failing). L.

### WP11. Pi rehearsal

- WP0 on the Pi with ME3 live; sample `vcgencmd get_throttled`, `free -m` and RSS each second.
- Also verify that `pi_speaker` streams sentences and how it handles the appended correction.
- **Pass:** first searchable under 20 s; ready under 3 min; search under 400 ms throughout; no OOM.
- Measures pdfium on the Pi.
- **Effort:** S.

## 2. Order and parallelism (rev 3)

```
WP0 ──(step 2 needs WP1's extractor)──► measures everything
WP1 ─► WP2 ─► WP3 ─► WP8 ─► WP9 ─► WP11
WP6 ────────► WP3                (doc_answer.norm overlap)
WP5 ─► WP7 ; WP5 ─► WP10(opt)
```

- **Wave 1, in parallel** (disjoint files): WP0, WP1, WP5, WP6. Measurement coupling: WP5 and WP6 both change the paired 112-question results. Run each one's A/B against the same base commit, then a combined run after both land.
- **Wave 2:** WP2 (needs WP1, shares `docs.py`); WP7 (needs WP5, shares the format tables).
- **Wave 3:** WP3, once WP2 and WP6 have landed and the WP0 questions are frozen.
- **Wave 4:** WP8, then WP9 (approval), then WP11.
- **Effort total (HYPOTHESIS):**
  - WP0 2-3 d, WP1 1.5-2 d, WP2 3-4 d, WP3 6-8 d, WP5 3 d, WP6 1 d, WP7 3 d, WP8 3-4 d, WP9 1 d, WP11 1 d.
  - About 25-31 dev-days in total.
  - With 3 parallel implementers, the critical path WP1 → WP2 → WP3 → WP8 → WP9 is about 15-19 working days.
  - Start WP3's Opus design during wave 1 to shorten it.
- **Cut order if short of time:** WP10, then WP7's xls/ODF, then WP8 stage 2. Never cut WP0, WP1, WP3, WP5, or WP8 stage 1a.

## 3. Owner decisions

1. **Checking mode:** route documents-first (raced, falling through silently) for every non-fast turn on the day?
2. **Live spend:** up to $0.05 for WP9? About $0.01 expected, or $0.016 with Q8 hedging.
3. **OCR:** detect-and-report only, or a 1-day Pi trial of rapidocr?
4. **Formats:** add `xlrd` and `striprtf` to `pi-rag`; skip `.doc` and `.ppt`?
5. **Limits:** raise 20 MB / 500 pages to 64 MB / 1,000 pages?
6. **Web source:** Gemini grounding only (no Tavily, per memory), or Tavily as well?
7. **Device:** checking on the Pi, or is the laptop allowed (hybrid retrieval: 98/112 vs 92/112)?
8. **Hedged document request:** two concurrent calls, roughly doubling the cost of the 1.31 s p90 call, to protect the 2.5 s budget?
9. **Spoken quotes:** speak the verbatim quote, or only "As per <file>: <answer>" with the quote shown in text? Three spoken citations with quotes run about 30 s.
10. **Verdict call (WP8 stage 1b):** speak `contradicted` corrections by default (recommended), and keep `off_question` (D2) as label-only, or speak a retraction for it as well?

## 4. Day-of dry run (`scripts/me4_rehearsal.py`, WP0)

```bash
export VOICE_AGENT_DOCS_FIRST=1
rm -rf ~/me4day && mkdir -p ~/me4day
uv run python scripts/me4_rehearsal.py --zip ~/checker.zip --corpus ~/me4day \
  --questions ~/me4day_questions.jsonl --mode offline --report /tmp/me4.json
# approved only: --mode live --budget-usd 0.05
# then 3 questions spoken end to end through agent_server + the real client
```

**Ten spot questions**, written on the day from the checker's files, with no tuning:
- a PDF paragraph;
- a PDF table value;
- a docx section;
- a pptx slide;
- a CSV/xlsx cell;
- an aggregate;
- a two-file question;
- a documents+web question;
- a question that names a file;
- an unanswerable question.

**Pass criteria:**
- Every member is `ok` or appears in `failures`. This includes:
  - a password PDF (`pdf is encrypted`, `docindex.py:1113,1182-1183`);
  - a scanned PDF (`no text layer` / WP7 status);
  - a 30 MB file (`too_large`, unless Q5 raises the cap);
  - files in nested folders (walker depth 20, `docindex.py:37`).
- A zip over 64 MB is accepted (WP1 archive limits).
- First searchable under 20 s; ready under 3 min (Pi).
- `search_docs` under 400 ms throughout.
- Single-fact questions: the gold file is in the passages and a cited answer is accepted.
- Two-file question: both files cited.
- Documents+web: a labelled web part.
- Unanswerable: abstain, with no quote.
- Nothing says "the documents don't say" before `ready`.
- Accepted document turns: first audio p95 under 2.5 s; `done` p95 under 4 s.
- Critique: no `contradicted` verdict left uncorrected.
- Two contradicting files: both values are spoken, from WP3e or the WP8 stage 1a+ conflict check.
- The weather question still gets its ME3 answer.

**What this run does NOT catch** (rehearse separately):
- STT mistranscription of file names and numbers (speak 5 questions aloud through the real microphone path);
- barge-in during a correction;
- venue Wi-Fi latency to Gemini;
- thermal throttling on a long first index (WP11);
- a checker's corpus much larger than mock-day;
- whether live Gemini actually emits `MISSING:` for a docs+web question (offline uses scripted replies; only WP9 shows it);
- aggregate questions: they still fail unless WP10 is built, and this is expected.

## 5. What I rank differently from `critic-opus.md`

- **Up:**
  - F5, now a *correctness* issue in docs-first mode and merged into WP3.
  - F6, which now includes xlsx (probe).
- **New at the top:**
  - the "never web_answer" prompt rule;
  - the exclusive race;
  - publish-at-end and build under lock;
  - the 2.5 s budget against multi-claim outputs.
- **Down:**
  - F7/F8, except through WP2(b);
  - F10, which is moot with atomic extraction;
  - F9 (a one-line `os.nice(10)` in the indexer; unmeasured).
- **Done:** F2, F3, and F1 on the server side. The 52 ms figure was measured on PDFs only.

## 6. Plan critique log (adversarial pass on rev 1, re-checked against code)

| # | Finding about rev 1 | What I checked | Verdict | Change |
|---|---|---|---|---|
| 1 | "Questions never reach the documents" was overstated | `agent_server.py:506-521`, `call_policy.py:34` | Partly wrong: routed `web_answer`/`wiki_summary` turns already race the documents | §0.1 narrowed to router replies and Gemini escalations |
| 2 | Docs-first via `handle_docs_turn` would break ME3 | `doc_route.py:341` (`NOT_FOUND_OFFER`) | Real regression: weather would get "the documents don't say" | WP3a: θ=0 implicit path, raced, silent fall-through; old WP4 merged |
| 3 | xlsx row labels: assumed fine / unverified | Probe: 41×61 sheet → 123 chunks, 75 without the row label | Confirmed bug | WP5 now covers xlsx; effort M → M+ |
| 4 | Parse-cache coupling with WP5 | `docindex.py:1934-1941` | Confirmed: parsed CSV text is cached | `LEGACY_PARSE_VERSION` bump is mandatory in WP5 |
| 5 | Old WP8 "names/numbers must be in own quote" as a reject rule | Compared with OnIt clearance (`verify.py:319-336`) | It would add false abstentions | Now a clearance rule (no call needed), not a rejection |
| 6 | Budget: multi-claim output plus web inside 2.5 s | `doc_answer.py:20`, `LATENCY.md:23-24`; request not hedged | Risk real; size HYPOTHESIS | Speak claims progressively; Q8 hedging; WP0 measures |
| 7 | A spoken multi-claim answer is too long | `format_citation` speaks the quote (`doc_route.py:126-141`) | Confirmed | WP3d voice rendering; Q9 |
| 8 | The existing background critic outlives the turn | `agent_server.py:662-706`, `self.checks` | Confirmed: no cancel on a new turn | WP8 stage 2 adds cancel and drops stale notices |
| 9 | WP3 effort 4-5 d too optimistic | Scope across `doc_answer`, `doc_route`, `agent_server`, prompt, benchmark | Optimistic | 6-8 d, with the critique split out to WP8 |
| 10 | WP2 effort "M" hides concurrency work | Shared FTS5 instance (`backends.py:73-76`) | Underestimated | M-L, 3-4 d; Opus critique |
| 11 | Hidden coupling: WP0 uses its own unzip | WP0/WP1 design | Rehearsal would test different code | WP0 step 2 must switch to WP1's extractor |
| 12 | Regression risk: `top_k` 8 adds distractors; tokenizer affects all rows | Not measurable now | Open | A/B `top_k` 4 vs 8; paired 112 + BEIR guards in WP3/WP6 |
| 13 | "Client speaks streamed sentences" stated from memory | `agent_client.py:12-16`, `doc_chat/__main__.py:886` | VERIFIED for `doc_chat`; `pi_speaker` not checked | WP11 verifies |
| 14 | "Gemini ~200 tok/s" used in my head, not in code | Nothing to read | Memory-only | Removed; WP0 measures output time |
| 15 | Report B cost figure; report E GIL claim; report C path claims | `LATENCY.md:24`; `docs.py:36,67`; `convert.py:732-753`, `docindex.py:1564` | B ~10-20x high; E right across processes but not within the docs server; C wrong twice | Kept in the package notes |
| 16 | The dry run cannot catch STT, barge-in, Wi-Fi, thermal | §4 | True | §4 lists separate rehearsals |

## 7. Final recheck log (cold reviewer pass on rev 2; 3 rounds)

**Round 1: 13 material findings.** Full cold read plus code checks.

| # | Finding | Check | Verdict | Change |
|---|---|---|---|---|
| R1.1 | The critique, if run inside the route, delays part of the spoken answer | `SentenceFilter` probe: the closing `"quote"` of a `format_citation` answer is held until `finish()` (`agent_server.py:956-960`) | Rev 2's claim "0 added latency" did NOT hold | The route returns a critique coroutine; `handle()` flushes first, then critiques, then sends `done`; new order test |
| R1.2 | The critique holds `self.lock`, so it blocks the next request | `agent_server.py:824` | Real | Cancel on a waiting request; label `preempted`; test |
| R1.3 | WP1 "reuse zipguard limits" would refuse real zips | `zipguard.py:13-16`: 64 MB total, 20 MB per member; ipo-2026 is 71 MB | Did NOT hold | Separate archive limits |
| R1.4 | WP8 claimed to fix D2 | The clearance rule clears a real but off-question quote | Did NOT hold | `off_question` label-only; D2 xfail stays; Q10 reworded |
| R1.5 | Stage-2 conflict notice gets killed by the new cancel-on-new-turn rule | Design self-contradiction | Real | Conflict check moved to stage 1a+ (local, 300 ms) |
| R1.6 | Stage 1a+ called `index.search` from the pipeline | The pipeline has no in-process index | Wrong | Use the `search_docs` MCP call |
| R1.7 | WP3a race assumed a `claim` hook | `routed_turn` has none (`agent_server.py:599-601`); only `run_routed` does | Plumbing missing | Risk noted in WP3a, counted in its effort |
| R1.8 | WP2d assumed the pipeline sees `indexing`/`ready` | `parse_docs_status` drops them (`doc_route.py:155-189`) | Gap | WP2d extends the parser |
| R1.9 | Per-claim judge feasibility | `quote_ok`/`names_numbers_ok` take plain strings (`doc_answer.py:241,248`) | Holds | Cited; the new work is the parser |
| R1.10 | Verdict 1.5 s vs `DOC_ANSWER_TIMEOUT_S` 2.5 s | `doc_route.py:329-330` wraps only `ask()` | Holds: the critique is outside the budget | Stated in WP8 |
| R1.11 | Dry-run walk: password PDF / scanned / 30 MB / nested folders / contradiction / file+web | `docindex.py:1113,1182-1183` (encrypted reported); 20 MB cap; depth 20; partly scanned pages skipped silently | Partial-scan silent skip and `MISSING` behaviour were uncovered | WP7 partial-scan status; §4 criteria and the "does not catch" list extended |
| R1.12 | Missing effort total; §1.0 referenced "§1.8"; §0.1 wording on router replies | Read | Contradictions | Effort total added; reference fixed; wording fixed |
| R1.13 | Acceptance tests runnable offline? | `bench_fakes.fake_doc_request` (`:261`), `scripts/experiments/beir_calibrate.py`, `mcp` 1.30 in root venv: all exist | Holds, except BEIR needs a one-time free download | Noted in WP6 |

Confirmed in round 1, no change needed:
- the routed lookup race (`agent_server.py:506-521`);
- the web second check and its missing cancel (`agent_server.py:662-706`; tasks only discarded when done);
- the xlsx probe (48 of 123 chunks with label, so 75 without).

**Round 2: 4 material findings.** Re-verified round 1's edits.

| # | Finding | Check | Verdict | Change |
|---|---|---|---|---|
| R2.1 | 9 `docindex.py` line cites were stale (tickets 21-22 shifted the file); also `docs.py:37,65` and the lock line | grep at HEAD | Wrong cites | Corrected: 44-45, 2225-2240, 2258, 1564-1565, 49-85, 317-340, 1125,1176, 1113,1182-1183; `docs.py:36,67`; `agent_server.py:824` |
| R2.2 | WP6 "bump `tokenize_fingerprint`" does nothing | `fts5_store.py:30-32,71-76`: an all-ASCII probe hashed through `tokenize` | Did NOT hold: stale on-disk FTS5 would survive | WP6 must add non-ASCII words to `_FINGERPRINT_PROBE` |
| R2.3 | WP5 and WP6 run in parallel but share the paired baseline | Design | Hidden coupling | Same-base A/B for each, then a combined run |
| R2.4 | WP8 heading still said it absorbs answerability | Read | Contradiction with R1.4 | Heading reworded |

**Round 3: 1 minor finding, nothing material.** Re-verified round 2's edits:
- every cited line was rechecked by grep;
- the order graph and waves match the dependencies;
- no references to removed packages remain;
- the only fix was the Q2 cost wording (now matches WP9).

Pass clean.

**Remaining open risks (not fixable by editing the plan):**
- the multi-claim latency inside 2.5 s (WP0 measures it);
- whether Gemini emits `MISSING:` reliably (WP9 only);
- D2 off-question answers;
- `pi_speaker` streaming and appended correction (WP11);
- aggregation without WP10;
- the effort total against the unknown check date.

