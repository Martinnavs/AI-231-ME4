# Ticket 273: MMLongBench-Doc on a 25-PDF corpus, larger dev run, 50-row calibration sheet

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). Builds on 272 (`2cbdaec`). Free: Qwen only, no Gemini/web.
`SUFFICIENCY` stays False. Commit with explicit paths only. Do not run the test split.

## Why
272's sample touches 110 PDFs (13.6k chunks in one DB); 15 of 20 dev questions got no answer from retrieval.
The official task is one PDF per question, and the Pi budget fits about 25 PDFs. Owner decided: baseline and A/B
use a ~25-PDF corpus.

## Do
1. Change `sample` to pick DOCUMENTS first, then questions:
   - Choose 25 PDFs with a seeded sha256 rule (like 272's), stratified so that text, table, layout and
     unanswerable-bearing documents are all present; prefer documents with >= 6 usable questions; cap pages so the
     total is about 1,200 pages (report the real total and the share of scanned pages <20 chars/page).
   - Then sample about 200 questions only from those 25 documents (all of them if fewer), keeping the 272 rules:
     unanswerable quota scaled to what exists (aim ~20%), vision-only rows flagged `out_of_scope_vision` and kept
     out of the headline, same dev/test hash split. Report the counts per stratum and split; if the pool is under
     200 say so and use what exists (do not pad).
   - Keep the old 110-PDF manifest as `eval/mmlongdoc/manifest-110.json` + `questions-110.jsonl` (rename, not
     delete); the new one is `manifest.json` / `questions.jsonl`. Nothing else reads the old names: grep first.
2. Re-run `fetch` (only needs the new PDFs) and `ingest` into a NEW db dir (`out/mmlongdoc/corpus25/`), same
   pdfium engine, no `--ocr`. Report file count, chunks, seconds, peak RSS.
3. `run --split dev` over ALL dev questions (about 100; resumable; ~1 s each), then `score`, then
   `calibration-sheet` which must now write 50 rows (stratified, seeded) to `out/mmlongdoc/calibration-50.md`.
   Do not grade the sheet.
4. Update `docs/MMLONGDOC-BENCH.md` (the 25-PDF corpus, the new counts, the old 110 manifest kept) and the tests
   for the document-first sampling (deterministic, 25 docs, all questions belong to chosen docs, split stable).

## Acceptance
- pytest for `tests/test_bench_mmlongdoc.py` passes; `ruff check` and `ruff format --check` clean on your files;
  the full root suite shows no new failures versus: 6 known (5 `test_me4_entry_eval`, 1
  `tests/doc_chat/test_flow.py::test_llm_echo_is_accepted_without_documents`) plus pipeline's
  `test_docs_wiring` one. Do not run the whole root suite more than once (20 min); run only the touched tests.
- Final message: counts (docs, pages, scanned share, questions per split/stratum/unanswerable/vision), ingest
  stats, dev score table (headline in-scope accuracy, unanswerable accuracy, by format, by evidence source, lane
  none vs accepted counts, cited-file == doc_id rate), the three worst failure patterns, anything not done.

## Non-goals
No change to ask_documents/agent_server/sufficiency, no OCR work (T5), no prompt tuning to raise the score, no
edits to the vendored official rules, no test-split run, no Gemini/web.
