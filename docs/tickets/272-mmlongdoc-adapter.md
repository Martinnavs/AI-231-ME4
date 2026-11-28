# Ticket 272 (ideation T3): MMLongBench-Doc adapter, offline runner and Qwen scorer bridge

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). `SUFFICIENCY` stays False. Free: Qwen only, NO Gemini,
NO web search, no paid call of any kind. Read `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md` (section 3, T3)
and `docs/tickets/269-ab-freeze-vs-sufficiency.md` first; this ticket is the producer for 269.

## Goal
Build the benchmark plumbing so that one command samples a pinned MMLongBench-Doc subset, ingests its PDFs,
answers each question through the REAL document lane with local Qwen as the answerer, and scores the answers
with the official rules and a Qwen answer extractor. Later arms (269) reuse it unchanged.

## Facts (verified by the orchestrator)
- HF dataset `yubo2333/MMLongBench-Doc`, revision `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`. Files:
  `data/train-00000-of-00001.parquet` (the questions), `documents/*.pdf` (135 PDFs). HF is reachable by curl.
  Download with `https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/resolve/<rev>/<path>`; only the PDFs
  the subset needs. Download to `~/bench_data/mmlongdoc/` (NOT in the repo; never commit PDFs or the parquet).
- Official rules: `https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/eval/eval_score.py`
  (and `extract_answer.py` for the GPT-4o extractor prompt). Vendor the rules file as
  `scripts/mmlongdoc_eval_score.py` with a header comment (source URL, date fetched 2026-11-28, Apache-2.0).
  Do not edit its scoring logic.
- Qwen relay: use the helpers in `scripts/sufficiency_eval.py` (`Relay`, `relay_payload`, `make_requests`,
  `TokenCounter`, the `Harness` pattern, `--today`). The lane is `voice_pipeline.ask_documents.ask_documents`
  (needs `now=`, a tz-aware datetime; use the Manila clock helper from 268). Mirror how sufficiency_eval opens
  the DB and builds `context` and `card`. Read `scripts/me4_ask_eval.py` for how a folder is ingested and asked.
- Ingest CLI: `src/voice_agent/rag/ingest.py` (`python -m voice_agent.rag.ingest <folder> ...`; read its
  `--help`). Ingest the unzipped PDF folder WITHOUT `--ocr` for this ticket (OCR is ticket T5).

## Deliverables
1. `scripts/bench_mmlongdoc.py` with subcommands:
   - `sample`: reads the parquet, filters to answerable-in-scope plus unanswerable, applies the seeded
     stratified sample of about 200 questions: strata by `doc_type`/`evidence_sources` (text, table, layout,
     chart/image) and about 40 `Not answerable`; chart/image-only questions are kept but flagged
     `out_of_scope_vision: true` and reported separately, never in the headline. Seeded sha256 rule like
     `implicit-split:7` (look at how the implicit set does it), split dev/test by the same rule (about 50/50).
     Writes `eval/mmlongdoc/manifest.json` (revision, row count of the parquet, seed, counts per stratum and per
     split, list of PDFs needed with sha256) and `eval/mmlongdoc/questions.jsonl` (id, doc_id, question, answer,
     answer_format, evidence_pages, evidence_sources, split, flags). Record the real parquet row count (the paper
     says 1,091, some sources 1,082) and the share of scanned pages (<20 characters per page via pypdf text
     extraction, or pdfminer if that is what the repo uses) in the sampled PDFs.
   - `fetch`: downloads the PDFs the manifest lists, verifies sha256 against the manifest.
   - `ingest`: builds ONE corpus DB for the sampled PDFs (unzip-style folder, no `--ocr`), records seconds and
     file count. Ingest runs on the folder, not a zip.
   - `run --split dev|test --limit N`: answers each question through `ask_documents` in docs-only mode (no web,
     no supplement, no fallback; record `status`, `spoken`, `file`, `quotes`, `cited_pages` if present on
     DocResult, calls, seconds). Per-question jsonl at `out/mmlongdoc/run-<tag>.jsonl`. Resumable: skip ids done.
     The question is asked as is, against the whole corpus DB (not one DB per PDF); also record whether the
     cited file equals the question's `doc_id`.
   - `score`: extracts a concise typed answer from `spoken` with Qwen (temperature 0, the official extractor
     prompt adapted, JSON out), then applies the vendored official rules to get per-row score, plus aggregate
     accuracy, F1 (unanswerable), accuracy by answer_format, by evidence source, by split, with
     `out_of_scope_vision` rows reported separately. Writes `out/mmlongdoc/score-<tag>.json` and a Markdown table.
   - `calibration-sheet`: writes 50 rows (stratified, seeded) with question, gold, spoken answer, Qwen's
     extracted answer and the rules' verdict to `out/mmlongdoc/calibration-50.md` for hand audit by the
     orchestrator. Do not grade them yourself.
2. Tests `tests/test_bench_mmlongdoc.py` (root `tests/`) and/or `pipeline/tests/`: sampling is deterministic and
   the split is stable; the manifest counts add up; the scorer wrapper gives the official score on a small fixed
   table of (gold, extracted) pairs covering Str/Int/Float/List/Not answerable; no network in tests (fake
   transport); runner skips done ids; status mapping.
3. A short `docs/MMLONGDOC-BENCH.md`: how to run each subcommand, where data lives, what is and is not measured
   (static docs: the sufficiency gain is NOT shown here; vision-only questions are out of scope; Qwen is the
   answerer and the extractor, so numbers are not comparable to published GPT-4o numbers).

## Acceptance
- `sample` writes the manifest and questions; counts shown (total, per split, per stratum, unanswerable, vision).
- `fetch` + `ingest` + `run --split dev --limit 20` + `score` all run end to end on the real data and print the
  table. Report the real numbers and the seconds in your final message. If the Qwen relay is slow, run 20 not more.
- Full `pytest` in `pipeline/` and root `tests/` unchanged except new tests (one known failure,
  `test_docs_wiring::test_relative_corpus_baby_onit_resolves_to_repo_root`, is pre-existing; leave it);
  `ruff check` and `ruff format --check` clean on your files.
- Nothing from `~/bench_data` or `out/` committed. Commit with explicit paths only (never `git add -A`).

## Non-goals
No Gemini or web calls, no changes to ask_documents/agent_server/sufficiency, no OCR fix (T5), no A/B, no tuning
prompts to raise the score, no editing the official scoring rules, no audio. Do not run the test split; dev only.

## Report
End with: manifest counts, scanned-page share, ingest seconds, dev 20-question score table, the three worst
failure patterns you saw, and anything you could not do. Do not claim agreement with the official GPT-4o scorer;
that is the 50-row calibration the orchestrator does next.
