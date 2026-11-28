# MMLongBench-Doc bench (ticket 272)

The offline MMLongBench-Doc adapter, runner and scorer bridge. Producer for the
ticket-269 A/B: one script samples a pinned dataset revision, ingests only the PDFs
the subset needs, answers each question through the **real document lane**
(`voice_pipeline.ask_documents.ask_documents`, docs-only) with the local Qwen relay
as the answerer, and scores with the **official rules** (vendored unchanged) after
a Qwen typed-answer extraction.

## Commands

All from the repo root. The `sample`/`fetch` steps need `pyarrow` (parquet) and
`pypdf` (scanned-page share), so they run under the pipeline project's `uv`
overlay; `ingest` needs the `pypdfium2` overlay (it parses the PDFs with the
repo's preferred pdfium engine); `run`/`score` work with the plain pipeline venv.
Nothing here costs money (local Qwen relay only).

```bash
cd pipeline

# 1. Sample: writes eval/mmlongdoc/manifest.json + eval/mmlongdoc/questions.jsonl
#    (downloads the 82 KB parquet to ~/bench_data/mmlongdoc/ if missing)
uv run --with pyarrow --with pypdf python ../scripts/bench_mmlongdoc.py sample

# 2. Fetch: downloads ONLY the PDFs the manifest lists, verifies each content hash
#    against the manifest (sha256 for LFS files, git blob sha1 for the rest), scans
#    the sampled PDFs (<20 chars/page via pypdf) and records the scanned-page share
#    back into the manifest. Re-runs skip verified files.
uv run --with pyarrow --with pypdf python ../scripts/bench_mmlongdoc.py fetch

# 3. Ingest: one corpus DB for the sampled PDFs (flat folder, NO --ocr; OCR is T5)
#    Engine: pdfium (the repo's preferred parser; `auto` picks it when pypdfium2 is
#    importable, else pypdf). Pinned here because pypdf alone cannot read 7
#    empty-password-encrypted 10-Ks and 1 token-limit PDF in this sample.
VOICE_AGENT_RAG_PDF_ENGINE=pdfium \
  uv run --with pypdfium2 python ../scripts/bench_mmlongdoc.py ingest
#    DB: out/mmlongdoc/corpus.db; re-run is incremental; --force rebuilds clean

# 4. Run: answer the split through ask_documents (no web, no supplement, no fallback)
uv run python ../scripts/bench_mmlongdoc.py run --split dev --limit 20 --tag dev20
#    resumable: ids already in out/mmlongdoc/run-dev20.jsonl are skipped

# 5. Score: Qwen extraction (official prompt adapted to JSON, temp 0) + official rules
uv run python ../scripts/bench_mmlongdoc.py score --tag dev20
#    writes out/mmlongdoc/score-dev20.json + score-dev20.md and prints the table

# 6. Calibration sheet: 50 stratified seeded rows (question, gold, spoken, Qwen's
#    extracted answer, the rules' verdict) for the orchestrator's hand audit
uv run python ../scripts/bench_mmlongdoc.py calibration-sheet --tag dev20
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
| manifest + sampled questions | `eval/mmlongdoc/manifest.json`, `eval/mmlongdoc/questions.jsonl` | yes |
| parquet, PDFs | `~/bench_data/mmlongdoc/` (`data/`, `documents/`) | never |
| corpus DB, flat corpus folder, run/score/calibration outputs | `out/mmlongdoc/` (`corpus.db`, `corpus/`, `run-*.jsonl`, `score-*.json`, `calibration-50.md`) | never |
| vendored official rules (Apache-2.0, logic unchanged) | `scripts/mmlongdoc_eval_score.py` | yes |

Dataset: HF `yubo2333/MMLongBench-Doc` @ `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`
(1,091 rows — the paper says 1,082; the revision is pinned and its row count +
parquet sha256 are recorded in the manifest). Official rules:
`github.com/mayubo2333/MMLongBench-Doc/eval/eval_score.py` (fetched 2026-11-28).

## The sample (frozen, seeded)

- ~200 questions: `unanswerable` (answer == "Not answerable", ~40, its own stratum
  with a fixed quota) + the rest stratified by **(evidence, doc_type)**; evidence
  priority text > table > layout; rows whose evidence is vision-only (Chart/Figure)
  are the `chart_image` stratum, **flagged `out_of_scope_vision`**, kept in the
  sample but never in the headline.
- Seeded sha256 rules (the `implicit-split:7` convention, own salts, seed `7`):
  - `qid` = `q<parquet_row_index:04d>`
  - within a stratum, rank = `int(sha256("mmlongdoc-sample:7:<qid>"), 16)`; the top
    quota rows per stratum are kept (largest-remainder quota, floor 1)
  - `split` = `dev` iff `int(sha256("mmlongdoc-split:7:<qid>"), 16) % 2 == 0` else `test`
- The manifest records: revision, parquet sha256 + row count, seed, the exact rule
  revision, counts per stratum / evidence / doc_type / answer_format / split, the
  needed PDFs with their content hash pinned from the tree API at that revision
  (LFS files: content sha256; plain git blobs: the git blob sha1), and the
  scanned-page share of the sampled PDFs.
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
- OCR is off (T5): scanned pages contribute little text; the scanned-page share
  is recorded in the manifest so that effect is visible.
- The corpus DB is built with the repo's **pdfium** PDF engine (`VOICE_AGENT_RAG_PDF_ENGINE=pdfium`,
  recorded in `out/mmlongdoc/ingest.json`): pypdf alone cannot read the
  empty-password-encrypted 10-Ks and one token-limit PDF in the sample. The 269
  arms must rebuild from the same raw files with the same engine.
- Docs-only mode: no web, no supplement, no fallback — the 269 arms change this
  (their agent paths run as in production).
- The official scoring rules are vendored verbatim (Apache-2.0, do not edit); the
  only adaptation is the extractor model/prompt-format (JSON out), which is why
  the 50-row calibration sheet exists before any number is trusted.
