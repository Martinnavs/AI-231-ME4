# Ticket 274: why does the lane give no answer on MMLongBench-Doc? (read-only diagnosis, free)

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). Builds on 273 (`5b11584`). Qwen only; no Gemini/web; do NOT
change ask_documents, agent_server, sufficiency, readers or ingest; no prompt tuning; dev split only.

## Question
On the 82 in-scope answerable dev rows the lane returned `none` (49) or `check_failed` (13) for 62 of them, versus
20 accepted. For each of the 62, which of these is the cause?
- **R0 question not self-contained:** the question only makes sense with the document in hand ("Which individual
  lead this program?", "what is the title?"). In a shared 25-PDF corpus no retriever can find the document. The
  official task gives one PDF per question. Decide with a Qwen yes/no judge on the question text alone, and also
  hand-readable (print the text).
- **R1 retrieval miss:** the gold evidence page of the gold document is NOT in the passages the lane was given
  (`context_fn(question)` as the runner calls it) -- check by page number via passage metadata and also by the gold
  answer string / its normalised number appearing in any passage of the gold document.
- **R2 page retrieved, answer not in text:** the evidence page is in the passages but the gold answer string does
  not appear in its extracted text (scanned page, chart/table the reader flattened, number split across lines).
- **R3 page retrieved, answer in text, lane declined:** the lane had the answer in its passages and still returned
  none / check_failed (strict acceptance, quote check, schema). For check_failed also record which check failed
  (read `DocResult.rejected`).
- **R4 other / cannot tell.**
Also record, for the 20 accepted rows, how many are correct, so the table shows the whole picture.

## Do
1. `scripts/bench_mmlongdoc_diagnose.py` (new; import helpers from `scripts/bench_mmlongdoc.py`, do not duplicate):
   for each in-scope dev row (not out_of_scope_vision, not unanswerable) rerun ONLY the retrieval step
   (`context_fn`) and the lane call as the runner does (resumable from `out/mmlongdoc/run-dev.jsonl` where possible;
   re-ask is fine, about 1 s each), and classify R0-R4 as above, writing `out/mmlongdoc/diagnose-dev.jsonl`
   (id, question, doc_id, evidence_pages, status, class, retrieved_pages_of_gold_doc, top files retrieved, gold
   answer found in gold-doc passages Y/N, rejected reason) and `out/mmlongdoc/diagnose-dev.md` (table: counts of each
   class overall and by evidence source and answer_format; the share of R1 whose gold page is a scanned page
   (<20 chars/page, from the manifest's `scanned_pages.per_pdf` or by recomputing); 5 example rows per class with
   question, gold, what was retrieved).
2. Also report: how many in-scope dev questions mention nothing document-specific (R0 count over all 82, not only
   the 62), because that bounds how well ANY shared-corpus run can do.
3. A small unit test with a fake context/lane for the classifier (one row per class). `ruff check` and
   `ruff format --check` clean on new files.
4. Commit with explicit paths only. Do not run the full root suite; run the new test file only.

## Final message
The class table, the R0 count over the 82, the scanned share of R1, the top 3 rejected reasons for check_failed,
and the 5 R3 rows (the most actionable ones) with question, gold, and what the lane's passages contained. State
plainly which classes you could not decide. Do not propose or make code fixes; just diagnose.
