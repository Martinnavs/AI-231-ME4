# Pi test report

**Date:** 2026-11-20. **Device:** Raspberry Pi 4 Model B Rev 1.2, 4 GB RAM (3,796 MB usable, 2 GB swap), Debian 13 (trixie), kernel 6.12.75+rpt-rpi-v8, aarch64. **uv** 0.13.4, **Python** 3.12.15 (uv-managed), **SQLite** 3.53.1 with `ENABLE_FTS5`.
**Commit tested:** `7205266` (tip of `agent-assessment`), copied to the Pi as a `git archive` (no `.git` on the Pi, no GitHub credentials there).
**Corpus:** `martinnavs/rag-mixed-docs-eval-2026` from Hugging Face, 61 files / 71 MB, in `~/rag-testsets/ipo-2026`.
**Conditions:** the owner's deployed ME3 stack (pi_speaker, agent_server, MCP server) was running on the same Pi during every measurement, so timings carry some load noise (load average 1.5 before step 4; no swap used, no OOM kill). Raw outputs are in `~/pi-rag-results/` on the Pi.

## Findings

| id | severity | evidence | suggested fix |
|---|---|---|---|
| F-html-charset | **real, medium; FIXED in the next commit** | `pi-rag` does not install `charset-normalizer`. `readers._decode_html_bytes` imports it for HTML with no declared charset and non-UTF-8 bytes; without it falls back to `decode("utf-8", errors="replace")`. On the Pi `tests/rag/test_readers_html.py::test_html_entities_and_non_utf8_latin1_meta_charset` fails: `'René' in 'Ren� met �lodie at the caf� in Z�rich.'`. Declared-charset HTML decodes correctly (lxml 6.1.3 / libxml2 2.14.6, same as the lock file). | add `charset-normalizer` to the `pi-rag` group, or a deterministic cp1252/latin-1 fallback in `_decode_html_bytes`. Not changed (handoff rule). |
| F-test-env | low | `test_rag_regress.py` (3 tests) pin `markitdown` and `pandas` versions and arms that are not installed; `test_canonical_index_cache.py::test_none_cache_dir_and_legacy_mode_do_not_write_cache` fails with `('memo.pdf', 'unsupported', 'markitdown not installed')`. The last one has no `importorskip`, so a Pi-like environment fails rather than skips. | `importorskip("markitdown")` on that test; skip the regress tests when the pinned packages are absent. |
| F-test | info (timing, data not defects) | `test_fts5_bench.py::test_fts5_5000_chunk_bench`: reopen 303.45 ms against a 250 ms bound. `test_canonical_security.py::test_linear_time_trimming_with_marker_and_2m_empty_lines`: 3.55 s against 3.0 s. Bounds not loosened. | none |
| F-latency | info | PDF conversion is the cost: `irs/i1040gi_form1040_instructions.pdf` 59.3 s, `raspberrypi/pico_datasheet.pdf` 19.0 s, `irs/p15_employers_tax_guide.pdf` 13.2 s (legacy pypdf, in process, no timeout). A first index of a folder with such PDFs is a minute-scale, inline operation; the cache makes later opens fast. | none (known: refresh runs inline, `docs/RAG.md` section 7) |
| F-server | info | Step 8 checked start and exit only; no MCP round trip (no client script, `fastmcp` not installed on the Pi). | owner decides whether to add a stdio round-trip script |

No F-install, F-fts5, F-determinism, F-memory or F-child findings.

## Measurements

**Step 0, integrity.** 44 files in the manifest, 44 on disk, 0 missing, 0 sha256 mismatches. No `.rag_cache` in the corpus folder.

**Install (H1).** `uv sync --group pi-rag`: 159.6 s, peak RSS 53 MB. Only the project itself was built; every dependency was a wheel. `.venv` 362 MB. `fastmcp`, `markitdown`, `pandas`, `magika`, `onnxruntime`, `torch`: none installed. (`/usr/bin/time` is not installed on the Pi and there is no sudo; wall time and peak RSS were taken with a small Python `resource` wrapper.)

**Step 1, probe on the 13 fixtures (H2).** `fts5_probe: ok`, `retriever_status: fts5:file`, `fallback: none`, `heavy_modules_loaded: none`. Import 377 ms / 29 MiB (x86 138 ms / 24 MiB); refresh of 13 files 2.04 s (x86 0.23 s); FTS5 build 18 ms; reopen-to-first-query 10.9 ms; query p50 0.33 ms; peak RSS 81 MiB. rank-bm25: build 1.05 s, p50 0.87 ms, RSS 63.6 MiB.

**Step 2, `tests/rag`.** 538 passed, 52 skipped, 7 failed, 3 errors in 169 s (peak 292 MB). Errors: `test_fts5_integration`, `test_fts5_wiring`, `test_rag_markdown_integration` (`No module named 'fastmcp'`, expected). `test_import_footprint` was not in the error or failure list (outcome not separately checked). Failures: see F-html-charset, F-test-env, F-test. The `tests/mcp_servers` suite was not run (all of it needs `fastmcp`); so the gemini-grounding keyring test is **not reported**.

**Step 3, fixtures, `--match tokens`.** FTS5 and rank-bm25 both: overall R@3 0.897, MRR 0.862 (x86 reference: R@3 0.897, MRR 0.862).

**Step 4, external set (112 answerable rows, FTS5 pretok, committed converters).**

| matcher | hits k=3 / k=4 | R@3 | MRR | wall | peak RSS (parent) | child peak RSS |
|---|---|---|---|---|---|---|
| tokens | 90 / 92 | 0.804 | 0.697 | 285.5 s | 268 MB | 221 MB |
| normalized | 78 / 80 | 0.696 | 0.603 | 284.2 s | 283 MB | |

x86 reference: tokens R@3 0.804 / MRR 0.697, normalized 80/112 at k=4, wall 69 s, RSS 252 MB. Pi wall is about 4.1x x86. All 44 files `ok`, 0 failed, no conversion timeout, no MemoryError; conversion 116.7 s of 146.5 s refresh. Per format R@3: pdf 0.750, docx 0.824, pptx 0.733, xlsx 0.895, html 1.000, csv 0.500, md 0.714, txt 1.000. Reported retriever string in this run: `fts5:memory` (the report label for a run without a persistent folder cache).

**Step 5, determinism (H3).** `rag_paired_compare.py --metric hit4` against the committed x86 per-query files: tokens 92/112 vs 92/112, normalized 80/112 vs 80/112, b=0, c=0, no flipped rows. In addition the `top` lists (sources and chunk ids) are identical on all 112 rows in both modes.

**Step 6, persistent index on the real folder (H4).** 44 files, 13,780 chunks. `docindex_refresh` 144.6 s (first build, includes conversion); `fts5_build_s` 7.84 s; DB 16.1 MiB; reopen-to-first-query 797 ms; query p50 5.17 ms, p95 19.4 ms; 10/10 queries non-empty; peak RSS 199 MiB (whole script 231 MB, 156 s). rank-bm25 on the same chunks: build 8.81 s, first query 96 ms, p50 99 ms, p95 115 ms, 231 MiB.

**Step 7, `convert_isolated` cold start, median of 3 (H5).** xlsx 1.00 s, pptx 0.78 s, html 0.39 s, all `ok` (x86: 0.95 / 0.62 / 0.30).

**Step 8, docs server (H6).** `python -m voice_agent.mcp_servers.docs --corpus <13 fixtures> --retriever fts5 < /dev/null`: exit 0, 2.9 s, peak RSS 78 MB, no output or traceback. This does not prove an answer round trip.

**Step 9, memory.** Lowest observed headroom was ~1.2 GB free / 2.35 GB available during and after the runs (sampled before and after, not continuously); swap unused; `dmesg` shows no OOM kill. Not sampled continuously.

**Step 10 (BM25 fallback).** Covered by the rank-bm25 runs in steps 1, 3 and 6; the `test_fts5.py -k fallback` tests were within the step 2 run (no failure in that file).

## Hypotheses

- H1 confirmed (wheels only, no sdist builds).
- H2 confirmed (`fts5_probe: ok`).
- H3 confirmed (identical hits and identical top-5 lists on 112 rows, both matchers).
- H4 confirmed with caveats: query p50 5 ms, reopen 0.8 s on 13,780 chunks, 199 MiB, 16 MB DB; first build of a PDF-heavy folder takes minutes.
- H5 confirmed on this corpus (no timeout or MemoryError; heaviest file 59 s in-process).
- H6 partly: starts cleanly in 2.9 s (under 10 s); round trip not tested, so inconclusive for "answers `search_docs`".

## Not run

`tests/mcp_servers` (needs `fastmcp`), an MCP stdio round trip, the agent-level path (`agent_server.py --docs-dir`), continuous memory sampling during step 4, anything needing API keys.

## Open questions for the owner

1. Add `charset-normalizer` to `pi-rag` (F-html-charset)?
2. Should the Pi-side first-index time (minutes for big PDFs, inline on the first question) get a background warm-up instead?

## Follow-up (same day)

F-html-charset: `charset-normalizer>=3.0` added to the `pi-rag` group (`pyproject.toml`, two lines in `uv.lock` edited by hand because the laptop's older `uv` rewrote unrelated lock entries). On the Pi: `uv sync --locked --group pi-rag` accepts the lock, installs `charset-normalizer==3.5.2` (wheel), `tests/rag/test_readers_html.py` 28 passed / 1 skipped (was 1 failure), and the full `tests/rag` run is 539 passed, 6 failed (the F-test-env and F-test cases above), 3 `fastmcp` errors. No heavy package was added (fastmcp, markitdown, pandas, torch, onnxruntime still absent). Disk cleanup done on the Pi at the owner's request: uv, pip and mozilla caches removed (+0.4 GB free, 3.0 GB now); the ME3 services and venvs were untouched and still running.

## Fresh-file latency, 10 repeats (Pi 4, 2026-11-20)

Script: warm persistent FTS5 index over 33 of the 44 corpus files (4,295 chunks, 40.0 s build), then for each of 11 held-out files, 10 times: remove it, refresh, wipe the canonical cache (so direct-reader formats do not get a cache hit), copy it back, time `DocIndex.refresh()` (parse + chunk + index; the next question's search is 9-15 ms on top). Production default converters. ME3 services were running; load average 0.01 before. Seconds; `extract` = `load_pdf_pages` / `decode_and_load` / `convert_isolated` (includes the isolated child's start-up for pptx/xlsx/html), `index` = `Retriever.build`, `chunk` = the remainder.

| file | KB | chunks | mean | median | sd | min | max | extract | chunk | index |
|---|---|---|---|---|---|---|---|---|---|---|
| md | 15 | 31 | 0.14 | 0.13 | 0.02 | 0.13 | 0.18 | 0.00 | 0.03 | 0.11 |
| docx | 34 | 19 | 0.17 | 0.17 | 0.01 | 0.17 | 0.21 | 0.04 | 0.03 | 0.10 |
| pdf, short | 84 | 7 | 0.37 | 0.36 | 0.02 | 0.36 | 0.41 | 0.25 | 0.03 | 0.09 |
| html | 193 | 11 | 0.60 | 0.60 | 0.01 | 0.59 | 0.63 | 0.47 | 0.03 | 0.10 |
| pptx | 47 | 3 | 0.92 | 0.92 | 0.00 | 0.91 | 0.93 | 0.79 | 0.03 | 0.09 |
| xlsx | 20 | 51 | 1.25 | 1.25 | 0.01 | 1.24 | 1.28 | 1.10 | 0.04 | 0.11 |
| pptx, 6 MB | 6074 | 78 | 1.75 | 1.75 | 0.02 | 1.73 | 1.79 | 1.54 | 0.10 | 0.12 |
| csv, 526 KB | 526 | 7,831 | 10.85 | 10.85 | 0.09 | 10.75 | 11.00 | 0.08 | 6.03 | 4.73 |
| pdf, IRS Pub 15 | 1674 | 456 | 13.68 | 13.69 | 0.08 | 13.54 | 13.79 | 12.86 | 0.25 | 0.57 |
| pdf, Pico datasheet | 17751 | 87 | 18.92 | 18.90 | 0.08 | 18.83 | 19.02 | 18.41 | 0.20 | 0.31 |
| pdf, IRS 1040 instr. (126 p) | 4331 | 911 | 58.88 | 58.87 | 0.09 | 58.72 | 59.06 | 57.41 | 0.53 | 0.93 |

Reading it (for optimisation on the cluster): run-to-run spread is under 1.5% (sd 0.00-0.09 s), so single runs were representative. PDF extraction (pypdf, in process) is 94-98% of the time of every large PDF, about 0.46 s/page on the IRS 1040 file; it has no timeout and no canonical cache (legacy mode). pptx/xlsx/html are dominated by the isolated child's start-up and import cost (0.5-1.1 s). The CSV is the only case where chunking (6.0 s) and index insert (4.7 s) dominate, 7,831 chunks from one file. Every refresh pays about 0.09-0.12 s in `Retriever.build` regardless of file size (a fixed diff over the whole chunk list); a no-change refresh is 21 ms.

## Answer-side numbers (offline, Pi 4)

The repo has no live-model answer numbers (generation runs on Gemini in the cloud; the live benchmark is not approved and needs keys). What can be compared offline is deterministic code: the `doc_answer` quote/abstention checks over the committed passage snapshot, and the offline agent benchmark with scripted model replies. Run on the Pi from a separate `pipeline/.venv` (`uv sync`, 437 s, 578 MB; pipecat 1.13.0, Python 3.12.15):

- `pipeline/tests` doc files (`test_doc_abstention`, `test_doc_answer`, `test_doc_gate_calibration`, `test_doc_route`, `test_rag_e2e`, `test_docs_wiring`): **141 passed, 1 xfailed** in 38 s. The xfail is the documented D2 defect (`test_rag_e2e.py::test_unanswerable_question_is_not_answered_from_an_irrelevant_real_sentence`).
- `python -m voice_pipeline.benchmark` (offline, 19 cases, 20 live-only skipped): abstention 16 pass / 0 fail / 3 n/a, `doc_citation` 4 / 0 / 15 n/a, `lookup_required` 16/0/3, `forbidden_absent` 15/0/4, `searched` 4/0/15, `duplicate_lookups` 6/1/12, `note_reached` 2/1/16, `support` 12/1/6, `source_coverage` 4/1/14. These equal the figures in `docs/AGENT-BENCHMARK.md` (abstention 16/0/3, doc_citation 4 pass and 15 n/a; the four failures are the four listed `known_failing` cases).

Why they cannot differ by platform: the snapshot and the retrieval it records are identical to x86 (step 5, all 112 top lists), and the checks are pure string code. What the Pi can change is time: the document race budget is 2.5 s, so a file that takes longer to read than that (every row above 1.75 s) is answered "still reading" on the question that follows its arrival. That behaviour is not covered by the offline cases and was not measured.

---

# Re-test, 2026-11-23 (commit `1e53a8e`, `origin/agent-assessment`)

Same Pi 4 (4 GB, Debian 13, aarch64, uv 0.13.4, Python 3.12.15, SQLite 3.53.1 with FTS5). Code sent as a `git archive` of `1e53a8e`; `uv sync --locked --group pi-rag` took 36.5 s and added `pypdfium2 5.15.2` (wheel), `xlrd 2.0.2`, `striprtf 0.0.35`; `charset-normalizer` was already in the group. The Pi had rebooted 8 minutes before the start; the ME3 stack was running again during every measurement. Raw outputs: `~/pi-rag-results/r2/` on the Pi.

## What could not be run

- **Private bundle** (`~/pi-bundle-me4/`: `mock2.zip`, `q.jsonl`, `q2.jsonl`, `tools/synth_corpus.py`, `tools/synth_check.py`): not on the laptop this session ran from, so handoff steps 4, 6 (offline rehearsal) and 8 (the owner's scale probe) were **not run as written**. Step 8 was replaced by a stand-in (below), so its numbers are not comparable with the owner's laptop reference.
- Step 3 and 7 (Gemini key, live rehearsal): no key on the Pi, owner decision, not run. Step 9 (`catdoc`): not installed, no sudo, not run. Step 10 (voice path through the deployed agent): not run, to leave the ME3 stack alone.
- E11 public-benchmark calibration (re-test item 2): not run (disk 2.4 GB free; FiQA alone is 782 MB on x86).
- A 400 MB scale point: not run (see below, swap was already used at 160 MB).

## Results

| check | Pi result | reference |
|---|---|---|
| probe on fixtures (`pi_rag_check`) | `fts5_probe: ok`, `fts5:file`, no fallback, no heavy modules; refresh of 13 files 0.72 s; import 1,165 ms (first run after a reboot) | first run: 2.04 s, 377 ms |
| `tests/rag` + `tests/test_me4_mockday.py` | 663 passed, 53 skipped, **6 failed, 3 errors** in 185 s | first run: 538 passed, 6 failed (after the charset fix), 3 errors |
| pipeline doc tests (6 files) | **173 passed, 1 xfailed** in 36.6 s (the documented D2 xfail) | first run: 141 passed, 1 xfailed |
| offline benchmark | identical to `docs/AGENT-BENCHMARK.md`: abstention 16/0/3, `doc_citation` 4/0/15, `lookup_required` 16/0/3, `forbidden_absent` 15/0/4, `searched` 4/0/15, `duplicate_lookups` 6/1/12, `note_reached` 2/1/16, `support` 12/1/6, `source_coverage` 4/1/14 | same as first run |

The 6 failures and 3 errors are the ones classified on 2026-11-20 (3 `fastmcp` imports; 3 `test_rag_regress` and the markitdown no-`importorskip` test need `markitdown`/`pandas`; two x86 timing bounds). No new failure appeared with the new code.

### External set (133 rows, 112 answerable; FTS5 pretok; same converter arm as before)

| engine / matcher | hits k=4 | R@3 | MRR | wall | peak RSS |
|---|---|---|---|---|---|
| pdfium (`auto`, new default) / tokens | 94 / 112 | 0.812 | 0.690 | 90.4 s | 207 MB |
| pdfium / normalized | 82 / 112 | 0.705 | 0.596 | 85.9 s | 211 MB |
| pypdf (`VOICE_AGENT_RAG_PDF_ENGINE=pypdf`) / tokens | 93 / 112 | 0.804 | 0.682 | 283.5 s | 248 MB |
| pypdf / normalized | 81 / 112 | 0.696 | 0.589 | 284.2 s | 256 MB |
| first Pi run (`7205266`) / tokens | 92 / 112 | 0.804 | 0.697 | 285.5 s | 268 MB |

Against the committed x86 references of `616c74a` (paired, `hit4`): pdfium gains 2 rows (`lex-2`, `tab-21`), loses none, p = 0.5, under both matchers; pypdf gains 1 (`tab-21`), loses none. PDF R@3 under tokens 0.750 -> 0.778. **The Pi is 3.1x faster end to end with the new PDF engine.**

**Determinism of the new code (H3 again).** Because the committed references predate the new code, the same commit and the same command were also run on the x86 laptop (isolated env from the lock file with the `pi-rag` group, 17 s / 15 s). Per-query comparison laptop vs Pi (pdfium): hits identical (94 and 82), `b=0, c=0`, and the top-5 source and chunk lists are identical on all 112 rows in both modes. (A first laptop attempt in the stale `.venv` read only 14 of 44 files because `python-pptx`, `lxml` and others were missing: an environment error, discarded.)

### Fresh-file latency, 10 repeats, pdfium default (same protocol as above)

Warm index of 33 of 44 files (4,440 chunks, 28.6 s build; it was 40.0 s before). Seconds, mean (sd), then the stage split extract / chunk / index.

| file | first re-test mean | previous mean | stages |
|---|---|---|---|
| md | 0.15 (0.02) | 0.14 | 0.00 / 0.03 / 0.11 |
| docx | 0.18 (0.01) | 0.17 | 0.04 / 0.03 / 0.11 |
| pdf, short | **0.15 (0.00)** | 0.37 | 0.02 / 0.03 / 0.10 |
| html | 0.61 (0.01) | 0.60 | 0.47 / 0.03 / 0.11 |
| pptx | 0.92 (0.01) | 0.92 | 0.79 / 0.03 / 0.10 |
| xlsx | 1.25 (0.01) | 1.25 | 1.09 / 0.04 / 0.11 |
| pptx, 6 MB | 1.75 (0.01) | 1.75 | 1.53 / 0.10 / 0.12 |
| csv, 526 KB | **7.60 (0.07)** | 10.85 | 0.24 / 3.14 / 4.21 |
| pdf, IRS Pub 15 | **1.97 (0.03)** | 13.68 | 0.88 / 0.26 / 0.83 |
| pdf, Pico datasheet | **1.08 (0.01)** | 18.92 | 0.55 / 0.20 / 0.33 |
| pdf, IRS 1040 instr. (126 p) | **4.93 (0.02)** | 58.88 | 3.01 / 0.55 / 1.37 |

pypdf on the same commit (3 repeats): 14.11 / 18.90 / 59.41 s for the three large PDFs, i.e. unchanged. **Pi factor of pdfium over pypdf: 6.9x (Pub 15), 17.5x (Pico), 11.9x (IRS 1040)** (x86 gain in E12: 15-23x). Extraction is no longer the whole cost for PDFs: the 0.1 s index floor and the chunking now matter. The CSV got faster (10.85 -> 7.60 s; chunk 6.0 -> 3.1 s) with the new table handling. A file arriving now is searchable within 2 s for everything except the large CSV (7.6 s) and the 126-page PDF (4.9 s). Every file up to 4.9 s is still above the 2.5 s per-question race budget only for those two.

### Scale probe (stand-in, not the owner's `synth_check.py`)

`synth_probe.py` (generates Zipf-vocabulary text, md and csv files plus unique "needle" facts, indexes the folder with `DocIndex` + FTS5 pretok and the default cache, measures time, peak RSS, cache size, search latency over the needles, needle hits at k=4, and a reopen). The same script was run on the x86 laptop; the Pi-to-laptop ratios are the comparable part. Single runs.

| raw size | machine | index | peak RSS (after reopen) | cache | search p50 / p95 | needles k=4 | reopen |
|---|---|---|---|---|---|---|---|
| 40 MB (79,309 chunks) | laptop | 14.2 s | 255 MB (289) | 159 MB | 0.4 / 0.5 ms | 15/15 | 6.1 s |
| 40 MB | **Pi** | **202.5 s** | 249 MB (286) | 159 MB | 2.6 / 3.1 ms | 15/15 | 31.0 s |
| 160 MB (313,202 chunks) | laptop | 61.1 s | 910 MB (1,036) | 623 MB | 0.6 / 0.7 ms | 57/57 | 24.7 s |
| 160 MB | **Pi** | **600.7 s (10 min)** | 895 MB (1,034) | 623 MB | 3.2 / 3.6 ms | 57/57 | 131.2 s |

Pi/laptop index time 14.3x at 40 MB and 9.8x at 160 MB (more than the 6x threshold in the handoff); memory and cache are the same as on x86 (about 5.6 MB of peak RSS and 3.9 MB of cache per MB of raw text for this synthetic corpus, 6.5 MB RSS after a reopen); retrieval stays exact on the needles; a restart costs 2 min at 160 MB because the reopen reads the cache again. A sampler logged memory every 2 s: lowest available memory 1,221 MB, **swap used up to 501 MB during the 160 MB run** (the probe was set to stop after a size that used swap, so 400 MB was not tried). Straight-line extrapolation (UNVERIFIED): 400 MB would need about 2.6 GB of RSS and 1.6 GB of cache beside the ME3 stack, and a 1 GB corpus about 6.5 GB of RSS and 3.9 GB of cache (does not fit RAM or the 2.4 GB of free disk), about 1 hour to index. The ceiling on this Pi with ME3 running is somewhere between 160 and 400 MB of raw text.

## Findings (new or changed)

| id | severity | evidence |
|---|---|---|
| F-scale | **high for a 1 GB corpus** | see the scale table: RAM per MB of raw text is the same as x86, but indexing is 10-14x slower, a restart re-reads the cache (131 s at 160 MB), and swap starts at 160 MB with ME3 running. |
| F-test-env (unchanged) | low | the markitdown test still has no `importorskip`; 3 regress tests need markitdown/pandas. |
| F-latency (resolved for PDFs) | info | pdfium cuts the three large PDFs from 13.7-58.9 s to 1.1-4.9 s on the Pi; CSV and 126-page PDF still exceed the 2.5 s race budget. |

## Open questions for the owner

1. Where is `~/pi-bundle-me4/`? Steps 4, 6 and 8 need it; the scale numbers above should be redone with `synth_check.py` to be comparable with the laptop reference.
2. Is the 1 GB corpus figure real? If so, the lazy or tiered index storage is needed; the Pi cannot hold the current design at that size.
3. Should the per-question budget (2.5 s) wait for a file that arrives with a read time of 5-8 s, now that PDFs are fast?
