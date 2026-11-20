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
