# RAG on the Raspberry Pi: test handoff

Written 2026-11-20 for the agent (or person) who tests the document-RAG component on a real Raspberry Pi 4.
Everything below was developed and measured on an x86 Linux cluster node. **No number in this repo has been measured on a Pi yet.** Your job is to measure it, find what breaks, and report back in the format at the end.

Read order: this file (what to run), then `docs/RAG-PI.md` (supported formats, install), `docs/RAG.md` (architecture), `docs/RAG-EXPERIMENTS.md` (what was measured and what is still open).

## 1. What is being tested

The RAG component lets a user drop files (md, txt, csv, pdf, docx, pptx, xlsx, html, epub) into a folder and ask the voice/text agent questions answered with cited verbatim quotes. On the Pi it must be lean:

- retrieval backend: SQLite FTS5 (stdlib `sqlite3`, tokenizer `pretok`), persistent index under `<docs-dir>/.rag_cache/fts5/index.sqlite3`; `rank-bm25` is the automatic fallback if FTS5 is missing;
- no embeddings on the Pi (no thin env, no ColBERT): `VOICE_AGENT_RAG_RETRIEVAL=auto` resolves to FTS5 when the embeddings environment is absent;
- converters: md/txt/csv/pdf/docx use the legacy in-process loaders (pypdf, stdlib zip/XML); pptx/xlsx/html/epub use the direct readers (python-pptx, openpyxl, lxml) which run in an isolated child process (30 s timeout, `RLIMIT_AS` 4096 MB, bare environment).

Hypotheses to confirm or refute on the Pi (all unverified):

| id | hypothesis | pass criterion |
|---|---|---|
| H1 | the `pi-rag` dependency group installs on aarch64 Python 3.12 without compiling (lxml, pillow, numpy wheels) | `uv sync --no-dev --group pi-rag` succeeds; wheels, not sdists |
| H2 | the Python used by uv has SQLite with FTS5 enabled | `scripts/pi_rag_check.py` prints `fts5_probe: ok` |
| H3 | retrieval is deterministic across platforms | per-query hits on the external set identical to the x86 run (92/112 at k=4, see section 5 step 5) |
| H4 | build/reopen/query latency and memory fit the Pi | see thresholds in section 5 |
| H5 | the 30 s child timeout and 4096 MB address-space limit are adequate on a Pi 4 for real documents | no `unreadable: conversion timed out` or MemoryError on the external corpus files; if there are, record file and time |
| H6 | the docs server starts and answers `search_docs` over stdio within a usable time | cold start < 10 s to first answer on the 13-file fixtures |

## 2. Prerequisites

- Raspberry Pi 4 (record RAM: 4 GB or 8 GB), 64-bit OS (aarch64), network for the first install only.
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`). The project needs Python >= 3.12; uv downloads a managed CPython 3.12 for aarch64 (it bundles its own SQLite, which is what FTS5 depends on).
- Git access to the branch: `origin/agent-assessment` (the owner pushes it; confirm `git log -1` shows the commit named in section 8).
- The external test corpus (not in the repo, 73 MB, 44 real public documents + 133-row golden set; `raw/`, `golden.jsonl`, `MANIFEST.json`, `AUDIT.md`, `tools/`). It is in a PRIVATE Hugging Face dataset, `martinnavs/rag-mixed-docs-eval-2026` (needs an `HF_TOKEN` with read access, set in the shell, never written to a file or committed). Download it to the exact path the commands below use:
  ```bash
  uv run python - <<'PY'
  import os
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id="martinnavs/rag-mixed-docs-eval-2026", repo_type="dataset", local_dir=os.path.expanduser("~/rag-testsets/ipo-2026"))
  PY
  ```
  (`huggingface_hub` is not in `pi-rag`; use `uv pip install huggingface_hub` in a scratch venv, or `scp -r <cluster>:~/rag-testsets/ipo-2026 ~/rag-testsets/` instead.) The folder name `ipo-2026` is historical. **Before Step 4 make sure `~/rag-testsets/ipo-2026/raw/.rag_cache` does not exist**: a cache written by an earlier run inside the corpus folder makes conversions cache hits and invalidates the timing measurements; delete it if present (the metrics CLI itself never writes there, but ad-hoc `DocIndex(folder)` scripts do). It is for local evaluation only: do not commit or redistribute it (university files are copyrighted; licences are listed in MANIFEST.json). Verify integrity: every file's sha256 must match MANIFEST.json (`python3 -c` snippet in section 5, step 0).
- Do NOT copy secrets, keyrings, `~/.secrets` or tokens to the Pi. The two files `pipeline/src/voice_pipeline/agent.py` and `src/voice_agent/mcp_servers/gemini_grounding.py` carry local-only keyring edits on the owner's cluster; on the Pi you work from the pushed branch where they are NOT modified. Never commit anything from the Pi without the owner's say-so.

## 3. Install

```bash
git clone <repo-url> AI-231-ME4 && cd AI-231-ME4 && git checkout agent-assessment     # or git pull
uv sync --group pi-rag                       # runtime + dev tools (pytest, ruff, mypy) + pi-rag packages
# runtime only (what the Pi would ship):  uv sync --no-dev --group pi-rag
uv run python -c "import sqlite3,sys;print(sys.version);print(sqlite3.sqlite_version)"
```

`pi-rag` = rank-bm25, pypdf, openpyxl, python-pptx, lxml, defusedxml (+ pillow, xlsxwriter, et-xmlfile through python-pptx/openpyxl). `fastmcp`, `markitdown`, `pandas`, `magika`, `onnxruntime`, `torch` must NOT be installed by this group: check with `uv pip list | grep -i -E "fastmcp|markitdown|pandas|magika|onnxruntime|torch"` (expect nothing). Record the install time and the disk footprint (`du -sh .venv`).

If any package builds from source (watch the output for "Building" / compiler invocations) or fails, record the package, version and error: that is finding F-install.

## 4. Known test-environment caveats (so you do not chase non-bugs)

- Several test files import `fastmcp` (laptop-only group) and will fail at collection on the Pi: `tests/mcp_servers/test_docs_server.py`, `test_docs_server_e2e.py`, `test_music_server.py`, `test_gemini_grounding.py`, `test_pages.py`, `test_web_info*.py`, and in `tests/rag`: `test_fts5_integration.py`, `test_fts5_wiring.py`, `test_import_footprint.py`, `test_rag_markdown_integration.py`. Run with `--continue-on-collection-errors`, or `--ignore` those files, and list which ones errored. Optionally install `fastmcp==2.14.7` into a scratch venv copy (`uv pip install fastmcp==2.14.7`) to run them, but say so in the report: it is not part of the Pi runtime.
- `tests/mcp_servers/test_gemini_grounding.py::test_without_a_gemini_key_tavily_is_used` fails on the owner's cluster only because of local keyring edits; on a clean Pi checkout it should pass or be skipped. Report either way.
- Tests that need MarkItDown, pandas or other laptop packages skip via `pytest.importorskip`; a skip is expected there.
- Timing tests (`tests/rag/test_canonical_spike.py` D5 ratios; `tests/rag/test_fts5_bench.py` FD6 bounds: reopen <= 250 ms, query p50 <= 50 ms, memory <= 150 MB and <= rank-bm25's, DB <= 40 MB, build <= 10 s at 5,000 chunks) encode x86 expectations. They MAY fail on a Pi because it is slower: that is data, not a defect. Record the measured values from `-s` output, do not loosen bounds.
- pypdf prints "fontTools is required to fully parse the encoding of a CFF Type1 font" warnings on some PDFs: harmless noise; filter with `2>&1 | grep -v -E "fontTools|Ignoring wrong pointing"`.
- A first run of anything that builds an index over the 44-file corpus is slow (conversion); the second run uses the canonical cache only for non-legacy formats.

## 5. Test plan (run in order; save all raw output under `~/pi-rag-results/`)

Create `mkdir -p ~/pi-rag-results` and prefix long commands with `/usr/bin/time -v -o ~/pi-rag-results/<name>.time` to capture wall time and peak RSS.

**Step 0. Corpus integrity.**
```bash
python3 - <<'EOF'
import hashlib, json, os
root = os.path.expanduser("~/rag-testsets/ipo-2026")
m = json.load(open(f"{root}/MANIFEST.json"))
files = m["files"] if isinstance(m, dict) and "files" in m else m
bad = 0
for f in files:
    p = os.path.join(root, "raw", f.get("path") or f.get("file") or f["name"]) if isinstance(f, dict) else None
    if p and os.path.exists(p):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != f.get("sha256"): bad += 1; print("MISMATCH", p)
print("checked", len(files), "mismatches", bad)
EOF
```
If MANIFEST.json uses other key names, adapt the snippet; the goal is sha256 equality for all 44 files. Mismatch = stop and report.

**Step 1. Environment probe (H1, H2).**
```bash
uv run python -I scripts/pi_rag_check.py --folder tests/fixtures/rag_corpus --compare-bm25 2>&1 | tee ~/pi-rag-results/probe_fixtures.txt
```
Expect `sqlite_compile_options` containing `ENABLE_FTS5=1`, `fts5_probe: ok`, `retriever_status: fts5:file`, `fallback: none`, `heavy_modules_loaded: none`. For reference the x86 run gave: import 138 ms / 24 MiB, refresh of 13 files 0.23 s, FTS5 build 5 ms, reopen-to-first-query 2.1 ms, query p50 0.07 ms, peak RSS 43 MiB; rank-bm25 build 197 ms, RSS 56.5 MiB. If `fts5_probe` fails the retriever falls back to rank-bm25 (the script says "fallback: rank-bm25"): record the exact reason, that is finding F-fts5.

**Step 2. Unit tests that can run on the Pi (H3 partially).**
```bash
uv run pytest tests/rag -q -p no:cacheprovider --continue-on-collection-errors 2>&1 | tail -n 60 | tee ~/pi-rag-results/pytest_rag.txt
```
x86 reference (laptop group installed): 1090 passed, 8 skipped, 3 xfailed with only the keyring test failing across `tests/rag tests/mcp_servers`. On the Pi expect many skips and the collection errors listed in section 4. Classify every non-pass into: expected (laptop-only import), timing bound (record values), or REAL FAILURE (report file::test and traceback tail).

**Step 3. Index and retrieval on the 13-file fixtures through the real entry point.**
```bash
uv run python -m voice_agent.rag.metrics --retriever fts5 --match tokens 2>&1 | tail -n 14
uv run python -m voice_agent.rag.metrics --retriever bm25 --match tokens 2>&1 | tail -n 14
```
x86 reference (29 answerable golden rows, exact matching, FTS5 pretok): R@3 0.897, MRR 0.862 for the default fixtures with the committed code; the two retrievers differ slightly. Any difference larger than one row from the x86 numbers for the SAME commit is a finding (non-determinism or platform bug).

**Step 4. The external set, end to end (H3, H4, H5).** This is the main measurement. It builds an index over 44 real files (13,780 chunks) through all converters, including the isolated child for pptx/xlsx/html.
```bash
cd ~/AI-231-ME4
CONV="pdf=legacy,docx=legacy,pptx=direct,xlsx=direct,html=direct,epub=direct"
/usr/bin/time -v -o ~/pi-rag-results/ext_fts5.time \
uv run python -m voice_agent.rag.metrics \
  --folder ~/rag-testsets/ipo-2026/raw --golden ~/rag-testsets/ipo-2026/golden.jsonl \
  --retriever fts5 --fts5-tokenizer pretok --converters "$CONV" --group-by type,format \
  --match tokens --report-json ~/pi-rag-results/ext_fts5.json --per-query ~/pi-rag-results/ext_fts5.jsonl \
  2>&1 | grep -v -E "fontTools|Ignoring wrong pointing" | tee ~/pi-rag-results/ext_fts5.txt | tail -n 40
```
Repeat with `--match normalized` (write `ext_fts5_norm.*`). x86 references (committed code `616c74a`, 112 answerable rows, FTS5 pretok):

| matcher | hits at k=3 | hits at k=4 | R@3 | MRR |
|---|---|---|---|---|
| `tokens` | 90 / 112 | 92 / 112 | 0.804 | 0.697 |
| `normalized` | 78 / 112 | 80 / 112 | 0.696 | 0.603 |

(`normalized` reads lower since the docx table change; see RAG-EXPERIMENTS.md E8: anchor-format bias.) x86 resources: wall 1:09 min, peak RSS 252 MB for the whole script (parent process; the child processes are separate). Pi expectation: same hits; wall several times longer. Record: wall time, peak RSS, and from `ext_fts5.json` the per-file diagnostic: any file with status other than `ok`, conversion seconds per file (the 126-page IRS PDF `irs/i1040gi_form1040_instructions.pdf` and the 18 MB `raspberrypi/pico_datasheet.pdf` are the heaviest; legacy pypdf runs in-process with no timeout), peak chunk counts.

**Step 5. Determinism check against x86 (H3).** The x86 per-query results for commit `616c74a` are committed in `docs/rag-reference/x86_ext_fts5_tokens_616c74a.jsonl` and `docs/rag-reference/x86_ext_fts5_normalized_616c74a.jsonl` (112 rows each: question, hit flags, top-5 sources and chunk ids). Compare:
```bash
uv run python scripts/rag_paired_compare.py docs/rag-reference/x86_ext_fts5_tokens_616c74a.jsonl ~/pi-rag-results/ext_fts5.jsonl --metric hit4
uv run python scripts/rag_paired_compare.py docs/rag-reference/x86_ext_fts5_normalized_616c74a.jsonl ~/pi-rag-results/ext_fts5_norm.jsonl --metric hit4
```
Expect `b=0, c=0` (identical hits). Any flipped row is a finding: send `id` and both top-5 lists (diff the `top` fields of the two JSONL rows; scores may differ in the last digits across platforms, source/id order must not).

**Step 6. Latency and memory of the persistent index on real data (H4).**
```bash
/usr/bin/time -v -o ~/pi-rag-results/pi_check_ext.time \
uv run python -I scripts/pi_rag_check.py --folder ~/rag-testsets/ipo-2026/raw --compare-bm25 \
  2>&1 | grep -v -E "fontTools|Ignoring wrong pointing" | tee ~/pi-rag-results/pi_check_ext.txt
```
Record `docindex_refresh` seconds, `fts5_build_s`, `reopen_to_first_query_ms`, `query_p50_ms`, `query_p95_ms`, `db_size_bytes`, `peak_rss`, and the bm25 comparison block. x86 reference for 5,000 synthetic chunks: FTS5 reopen 68 ms, p50 9 ms, ~54 MB, DB 5 MB; rank-bm25 rebuild 0.5 s, p50 10 ms, 127 MB. Note: the real corpus has 13,780 chunks, so numbers are not directly comparable; compare Pi vs x86 on the same folder (ask the owner for the x86 report of this exact command if needed, or run it there).

**Step 7. Child-process cost for the direct readers (H5).** Cold start of `convert_isolated` is the Pi's likely bottleneck (x86: xlsx 0.95 s, pptx 0.62 s, html 0.30 s per small file). Run:
```bash
uv run python - <<'EOF'
import statistics, time
from pathlib import Path
from voice_agent.rag.convert import convert_isolated
root = Path("tests/fixtures")
cases = {"xlsx": root/"rag_office/team_budget.xlsx", "pptx": root/"rag_office/site_review.pptx"}
html = b"<html><body><main><h1>T</h1><p>Hello world</p></main></body></html>"
for kind, p in list(cases.items()) + [("html", None)]:
    raw = p.read_bytes() if p else html
    ts = []
    for _ in range(3):
        t = time.perf_counter(); r = convert_isolated(raw, kind, engine="direct"); ts.append(time.perf_counter() - t)
    print(kind, r.status, "median_s=%.2f" % statistics.median(ts), [round(x, 2) for x in ts])
EOF
```
Pass: status `ok`; record the medians. If any is `unreadable` with "conversion timed out" or a MemoryError text, record the exact error (RLIMIT_AS and the 30 s timeout are constants in `src/voice_agent/rag/convert.py`; do NOT edit them, report).

**Step 8. The docs MCP server end to end (H6).** The server speaks MCP over stdio. A quick manual check:
```bash
mkdir -p /tmp/piddocs && cp tests/fixtures/rag_corpus/* /tmp/piddocs/ 2>/dev/null
VOICE_AGENT_RAG_RETRIEVAL=fts5 uv run python -m voice_agent.mcp_servers.docs --corpus /tmp/piddocs --retriever fts5 < /dev/null
```
It should start, build the index and exit on EOF without a traceback (record start-up time with `time`). For a real round trip use the MCP Python client (needs `fastmcp` or the base `mcp` package: `from mcp.client.stdio import stdio_client`); if you write such a script, keep it under `scripts/` and say so. The agent-level path is `pipeline/src/voice_pipeline/agent_server.py --docs-dir DIR --docs-retriever fts5` and needs the full pipeline dependencies: out of scope unless the owner asks.

**Step 9. Memory pressure and limits (H5).** On a 4 GB Pi run Step 4 while watching `free -m` and `dmesg | tail` for OOM kills; record the lowest free memory. If the build is OOM-killed, note at which file (the report JSON is not written then: use `--report-json` plus `tail` of stdout).

**Step 10 (optional). BM25 fallback path.** Force it and confirm identical behaviour: `VOICE_AGENT_RAG_RETRIEVAL=bm25` with Step 3's command, and simulate a missing FTS5 probe by running the unit tests in `tests/rag/test_fts5.py -k fallback`.

## 6. What not to change

- Do not edit source under `src/` or `pipeline/` to "make it work" on the Pi without the owner: report the failure with evidence. If a fix is obvious (a missing dependency in `pi-rag`, a path assumption) describe it precisely; the owner applies it through the ticket process.
- Never edit `pyproject.toml`/`uv.lock` on the Pi; report the package that is missing.
- Do not run anything that needs API keys, network search or the speech stack.
- Do not push from the Pi.

## 7. Report format (paste into `docs/RAG-PI-RESULTS.md` on a branch you create, or send to the owner)

```
# Pi test report (date, Pi model, RAM, OS, kernel, uv version, Python/SQLite versions)
Commit tested: <git rev-parse HEAD>
## Findings (each: id, severity, evidence, suggested fix)
F-install / F-fts5 / F-test / F-determinism / F-latency / F-memory / F-child / F-server ...
## Measurements (tables; raw files attached under pi-rag-results/)
Step 1 probe | Step 3 fixtures | Step 4 external (tokens + normalized) | Step 5 determinism | Step 6 latency | Step 7 child cost | Step 8 server start | Step 9 memory
## Hypotheses H1..H6: confirmed / refuted / inconclusive, one line each
## Open questions for the owner
```
Be explicit about what you did NOT run and why. Report numbers, not impressions.

## 8. State of the code you are testing

Branch `agent-assessment`. Latest commits (newest first): `616c74a` spike 2a (docx tables, html over-pruning, `--match tokens`), `124376e` spike 1 (soft/hard per-file cap, number tokenizer, zip media rule, `--per-query`, paired compare script), `796d00f`..`3415a81` the canonical pipeline / FTS5 / evaluation / docs commits. Plus this documentation commit. If the owner pushed newer commits, test the tip and say which commit.

Key facts to keep in mind while interpreting results:
- laptop vs Pi retrieval: the laptop uses hybrid (BM25 + ColBERT late interaction) when its embeddings environment exists; the Pi uses FTS5 alone. On the external set hybrid is better by about 6 rows of 112 at k=4 (mostly paraphrase), see RAG-EXPERIMENTS.md E10. The Pi will not reach the hybrid numbers; that is expected.
- all x86 timings were measured on a shared 256-core cluster node under varying load (load average 3-33): treat them as order-of-magnitude, the retrieval hit counts are exact and deterministic.
- the retrieval metric you run has two matchers: `normalized` (strict, penalises better table rendering) and `tokens` (converter-independent). Report both.

## 9. Ideas the owner may ask you to test next (do not start without being asked)

1. A Pi-sized semantic arm: a static-embedding model (model2vec / potion family, numpy lookup, tens of MB) fused with FTS5 by reciprocal rank fusion, measured paired against hybrid (see RAG-EXPERIMENTS.md section "Open").
2. PDF chunk structure (table-header repetition, font-size heading detection) for the numeric-table misses.
3. Raising or lowering `MAX_CHUNKS_PER_FILE`-style constants is NOT worth testing: the sweep is flat (E9c).
