# Ticket 275: is the text extraction the problem? (direct check, read-only, free)

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). Builds on 274 (`24e468c`) and P6. Qwen only; no Gemini/web; do NOT
change readers, ingest, ask_documents or any product code; dev split only (rows with `split == dev`); no test split.

## Question
Is the text we extract from the 25 corpus PDFs the reason answers are missed? Settle it without running the lane:
for each in-scope answerable dev question (not `out_of_scope_vision`, not unanswerable), is the gold answer actually
in the extracted text, and where it is not, why?

## Do
1. `scripts/bench_mmlongdoc_extraction.py` (new; reuse helpers from `scripts/bench_mmlongdoc.py`, no copies):
   a. **Per-PDF health** from `out/mmlongdoc/corpus25/corpus.db`: chunks, characters, pages with text vs the
      manifest's page count (`scanned_pages.per_pdf`), pages with no chunk. List every PDF with 0 chunks or under
      200 characters per page on average.
   b. **Gold-in-text, per row**: normalise gold (strip `$ , % and`, thousands separators incl. Indian `3,02,16,492`,
      case, "fiscal"/"FY" forms, list gold = every item) and check whether it appears (1) anywhere in the gold document's
      chunks in the DB; (2) on the gold `evidence_pages` pages specifically (page metadata); (3) in an INDEPENDENT
      extraction of the gold PDF's gold pages with a second extractor available in the venv (try pypdf, pdfminer.six,
      and pdfium via `pypdfium2`; use whichever are importable and say which; do not install anything). Classes:
      FOUND-BOTH, FOUND-INDEPENDENT-ONLY (our reader loses it), FOUND-DB-ONLY, ABSENT-BOTH (derived/computed or
      figure-only), DOC-EMPTY.
   c. **Table integrity** for rows whose evidence source is Table: for rows where the gold is a computed ratio or
      percentage, find the inputs the question names (e.g. the line items) and check that each input label and its
      number sit within the SAME chunk (a label and its value split across chunks is a structure loss). Report
      how many of the table rows pass, and show 5 examples of the extracted chunk text for a passing and a failing row
      (verbatim, so a human can read how the table looks after extraction).
   d. **The two empty slide decks**: find out why they produced no chunks. Read `out/mmlongdoc/ingest.json`/logs and run
      the three independent extractors on page 1-3 of each PDF; say whether the PDF has a text layer, is image-only,
      or the reader errored. Report what each extractor returns (first 300 chars) so a human can judge.
2. Output `out/mmlongdoc/extraction-dev.jsonl` and `out/mmlongdoc/extraction-dev.md`: the class counts overall and by
   evidence source and answer_format; the per-PDF health table; the table-integrity result with the examples; the
   slide-deck finding.
3. Tests with tiny fixtures (fake DB rows / a generated one-page PDF): normalisation (Indian grouping, $, %, FY),
   class assignment, per-PDF health. `ruff check` and `ruff format --check` clean; run the new test file only.
4. Commit with explicit paths only (script and test). Do not commit `out/`.

## Final message
The class table, the per-PDF health list (zero or thin PDFs), the table-integrity pass rate and two verbatim chunk
examples, the slide-deck verdict, and a plain statement: given these numbers, is extraction a primary cause of the
missed rows, a minor cause, or not a cause? Name what you could not check. No fixes, no proposals to change code.

## Non-goals
No reader or ingest change, no OCR run, no Gemini, no test split, no tuning.
