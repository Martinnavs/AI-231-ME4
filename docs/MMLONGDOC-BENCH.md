# MMLongBench-Doc bench (tickets 272/273)

The offline MMLongBench-Doc adapter, runner and scorer bridge. Producer for the
ticket-269 A/B: one script samples a pinned dataset revision, ingests only the PDFs
the subset needs, answers each question through the **real document lane**
(`voice_pipeline.ask_documents.ask_documents`, docs-only) with the local Qwen relay
as the answerer, and scores with the **official rules** (vendored unchanged) after
a Qwen typed-answer extraction.

Since ticket 273 the sample is **document-first**: the official task is one PDF per
question, so 25 documents are chosen first and ~200 questions are sampled from
those documents only. The 272 110-PDF sample is kept as `manifest-110.json` /
`questions-110.jsonl` (renamed, not deleted); nothing reads those names.

## Commands

All from the repo root. The `sample`/`fetch` steps need `pyarrow` (parquet) and
`pypdf` (page counts / scanned-page share), so they run under the pipeline project's
`uv` overlay; `ingest` needs the `pypdfium2` overlay (it parses the PDFs with the
repo's preferred pdfium engine); `run`/`score` work with the plain pipeline venv.
Nothing here costs money (local Qwen relay only).

```bash
cd pipeline

# 1. Sample (document-first, ticket 273): writes eval/mmlongdoc/manifest.json +
#    eval/mmlongdoc/questions.jsonl (the 25-doc corpus)
#    - picks 25 documents: ranked (prefer >=6 usable questions, then the seeded
#      sha256 doc rank, then doc_id); the 1,200-page cap swaps the largest chosen
#      doc for the smallest not-chosen one (>=6-usable tier first) while over
#      budget, keeping text/table/layout/unanswerable-bearing coverage
#    - then samples 200 questions from those documents only (the 272 strata,
#      the scaled ~20% unanswerable quota, and the frozen dev/test hash split)
#    (downloads the 82 KB parquet to ~/bench_data/mmlongdoc/ if missing)
uv run --with pyarrow --with pypdf python ../scripts/bench_mmlongdoc.py sample

# 2. Fetch: downloads ONLY the PDFs the manifest lists (usually none — sample
#    already fetched the page-count candidates), verifies each content hash
#    against the manifest (sha256 for LFS files, git blob sha1 for the rest),
#    scans the 25 PDFs (<20 chars/page via pypdf) and records the scanned-page
#    share back into the manifest. Re-runs skip verified files.
uv run --with pyarrow --with pypdf python ../scripts/bench_mmlongdoc.py fetch

# 3. Ingest: one corpus DB for the 25 PDFs (flat folder, NO --ocr; OCR is T5)
#    Engine: pdfium (the repo's preferred parser; `auto` picks it when pypdfium2 is
#    importable, else pypdf). Pinned here because pypdf alone cannot read 7
#    empty-password-encrypted 10-Ks and 1 token-limit PDF in this sample.
#    DB: out/mmlongdoc/corpus25/corpus.db (the default); the 272 110-doc DB stays
#    at out/mmlongdoc/corpus.db (explicit --db/--corpus-dir to use it).
#    Re-run is incremental; --force rebuilds clean. The ingest record is written
#    next to the DB (out/mmlongdoc/corpus25/ingest.json).
VOICE_AGENT_RAG_PDF_ENGINE=pdfium \
  uv run --with pypdfium2 python ../scripts/bench_mmlongdoc.py ingest

# 4. Run: answer the WHOLE split through ask_documents (no web, no supplement,
#    no fallback) - ticket 273 runs all ~100 dev questions, no --limit
uv run python ../scripts/bench_mmlongdoc.py run --split dev --tag dev
#    resumable: ids already in out/mmlongdoc/run-dev.jsonl are skipped

# 5. Score: Qwen extraction (official prompt adapted to JSON, temp 0) + official rules
uv run python ../scripts/bench_mmlongdoc.py score --tag dev
#    writes out/mmlongdoc/score-dev.json + score-dev.md and prints the table

# 6. Calibration sheet: 50 stratified seeded rows (question, gold, spoken, Qwen's
#    extracted answer, the rules' verdict) for the orchestrator's hand audit
uv run python ../scripts/bench_mmlongdoc.py calibration-sheet --tag dev
#    writes out/mmlongdoc/calibration-50.md - the script does NOT grade them
```

Defaults: relay `http://127.0.0.1:18080/v1`, model `Qwen/Qwen3.8-27B`
(`JUDGE_BASE_URL`/`--base-url` to override; the key comes from the
`judge_eval.read_key` convention: `JUDGE_API_KEY` or `~/.secrets/api_key.txt`).
`--today` (run) is the fixed date the lane's clock sees (default 2026-11-28, the
Manila clock helper from ticket 268).

## Where things live

| what | where | committed? |
| :--- | :--- | :--- |
| manifest + sampled questions (the 25-doc corpus, ticket 273) | `eval/mmlongdoc/manifest.json`, `eval/mmlongdoc/questions.jsonl` | yes |
| the 272 110-PDF sample (kept, untouched) | `eval/mmlongdoc/manifest-110.json`, `eval/mmlongdoc/questions-110.jsonl` | yes |
| parquet, PDFs | `~/bench_data/mmlongdoc/` (`data/`, `documents/`) | never |
| corpus DBs, flat corpus folders, run/score/calibration outputs | `out/mmlongdoc/` — the 25-doc one: `corpus25/corpus.db` + `corpus25/corpus/` + `corpus25/ingest.json`; the 272 110-doc one: `corpus.db` + `corpus/`; plus `run-*.jsonl`, `score-*.json`, `calibration-50.md` | never |
| vendored official rules (Apache-2.0, logic unchanged) | `scripts/mmlongdoc_eval_score.py` | yes |

Dataset: HF `yubo2333/MMLongBench-Doc` @ `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`
(1,091 rows, 135 documents — the paper says 1,082; the revision is pinned and its
row count + parquet sha256 are recorded in the manifest). Official rules:
`github.com/mayubo2333/MMLongBench-Doc/eval/eval_score.py` (fetched 2026-11-28).

## The sample (frozen, seeded, document-first since ticket 273)

**Documents first** (seed `7`, the `implicit-split:7` convention, own salts):

- `doc_rank(doc_id)` = `int(sha256("mmlongdoc-doc:7:<doc_id>"), 16)`; candidates
  are ranked by (prefer >=6 **usable** questions, then `doc_rank`, then doc_id)
  and the top 25 are taken. "Usable" = answerable and not vision-only.
- **Page cap (~1,200):** while the chosen documents' pages exceed the budget, the
  largest chosen doc is swapped for the smallest not-chosen doc (the >=6-usable
  tier first) when the swap lowers the total and keeps every evidence capability
  (text/table/layout/unanswerable-bearing) the dataset carries. Page counts are
  the pinned PDFs' real counts (local, or fetched free for the >=6-usable tier
  at sample time) with a deterministic parquet fallback (max evidence page),
  recorded per doc as `page_source`.
- The 25 chosen documents cover text, table, layout and unanswerable-bearing
  evidence (the manifest's `docs` list records pages, question counts and caps).

**Then questions, only from those 25 documents** (the 272 rules, unchanged):

- ~200 questions (all of them if the pool is smaller — no padding):
  `unanswerable` (answer == "Not answerable") is its own stratum with a **scaled**
  quota `min(pool unanswerable, round(total * 0.2))` (~20% of what exists), the
  rest stratified by **(evidence, doc_type)** with evidence priority text >
  table > layout; rows whose evidence is vision-only (Chart/Figure) are the
  `chart_image` stratum, **flagged `out_of_scope_vision`**, kept in the sample
  but never in the headline.
- Within a stratum, rank = `int(sha256("mmlongdoc-sample:7:<qid>"), 16)`; the top
  quota rows per stratum are kept (largest-remainder quota, floor 1).
- `split` = `dev` iff `int(sha256("mmlongdoc-split:7:<qid>"), 16) % 2 == 0` else
  `test` — the same frozen per-qid hash as 272, so a question's split never
  depends on which documents were chosen.

**The frozen 273 numbers** (seed 7; in `manifest.json`): 25 documents, 1,039
pages (the 1,200 budget bound — one 196-page doc swapped for an 18-page one),
242/1,039 = 23.3% scanned pages (<20 chars/page); the question pool of those 25
documents has 263 questions (41 unanswerable) — over 200, so 200 are sampled:
dev=109, test=91; unanswerable=40 (20%); in-scope answerable=151;
out-of-scope vision=9; by evidence text=74, table=54, layout=16, chart_image=16.

The 272 sample (110 PDFs, 5,312 pages, 200 questions) is kept verbatim as
`manifest-110.json` / `questions-110.jsonl`.

- Tune/debug on **dev only**. The test half is for the 269 arms.

## What is and is NOT measured here

**Measured:** the real document lane end to end (retrieval -> Qwen JSON answer ->
quote checks -> spoken form) on static long documents: in-scope accuracy
(text/table/layout evidence), unanswerable accuracy + the official unanswerable-aware
F1, by answer_format / evidence source / split, cited-file match, and the
out-of-scope vision rows reported separately.

**NOT measured / caveats (honest framing for 269):**
- This is a **static-document** benchmark: almost nothing is stale, so the
  **sufficiency gain is NOT shown here**. 269 measures no-regression +
  over-trigger cost; the gain lives on the 100-row sufficiency set.
- **Vision-only questions (Chart/Figure evidence) are out of scope** for the
  text-only lane (no VLM). They are kept as a flagged control, reported
  separately, never in the headline.
- **Qwen is both the answerer and the extractor** (the official pipeline uses a
  different model to answer and GPT-4o to extract). Numbers are therefore
  **not comparable to the published GPT-4o numbers**; they are comparable across
  269's arms (same answerer, extractor, rules, corpus).
- OCR is off (T5): scanned pages contribute little text; 23.3% of the 25-doc
  corpus's pages are scanned, and the manifest records the share.
- The corpus DB is built with the repo's **pdfium** PDF engine
  (`VOICE_AGENT_RAG_PDF_ENGINE=pdfium`, recorded in
  `out/mmlongdoc/corpus25/ingest.json`): pypdf alone cannot read the
  empty-password-encrypted 10-Ks and one token-limit PDF in the sample. The 269
  arms must rebuild from the same raw files with the same engine.
- One PDF per question is the official setup, and the 25-doc corpus keeps it
  faithful: retrieval runs over the whole 25-doc DB, and the `cited file ==
  doc_id` rate measures how often the lane finds the right document.
- Docs-only mode: no web, no supplement, no fallback — the 269 arms change this
  (their agent paths run as in production).
- The official scoring rules are vendored verbatim (Apache-2.0, do not edit); the
  only adaptation is the extractor model/prompt-format (JSON out), which is why
  the 50-row calibration sheet exists before any number is trusted.
