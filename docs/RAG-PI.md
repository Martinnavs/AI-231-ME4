# RAG on the Raspberry Pi

Written 2026-11-19, uncommitted work on branch `local/agent-assessment`. The Pi runs the retrieval half of `docs/RAG.md`: SQLite FTS5 (or `rank-bm25`) over text it can extract with pure-Python packages (direct readers for office, html and epub files). It has no embedding model, no MarkItDown, no pandas and no magika. Numbers for a real Pi 4 are not measured yet (the direct readers were measured on x86 only: external set, 64 s wall and 221 MB peak over the 44-file folder, see `docs/RAG.md` section 5); `scripts/pi_rag_check.py` produces them.

## Install

```
uv sync --no-dev --group pi-rag
```

The `pi-rag` group (`pyproject.toml`) holds seven packages on top of the base dependencies: `rank-bm25` (fallback retriever), `pypdf` (PDF text), `openpyxl` (xlsx), `python-pptx` (pptx), `lxml` (html, epub), `defusedxml` and `charset-normalizer` (HTML with no declared charset). The `pi` group is the audio check only and does not include them; the `laptop` group adds MarkItDown and is not for the Pi.

## What works

| Format | On the Pi | Why |
|---|---|---|
| `.md`, `.txt`, `.csv` | yes | read as text |
| `.pdf` | yes (needs `pypdf`, in `pi-rag`) | legacy loader, page by page; no OCR |
| `.docx` | yes | legacy loader, zip-guarded |
| `.pptx`, `.xlsx`, `.html`/`.htm`, `.epub` | yes (packages in `pi-rag`) | direct readers (`src/voice_agent/rag/readers.py`) in the isolated child process; a missing package gives `unsupported (<package> not installed)`. MarkItDown is an optional arm, not installed on the Pi |
| Hybrid / `auto` with embeddings | no | the sidecar needs the thin env (`docs/RAG.md` section 3); without `~/.venvs/rag-embed`, `auto` resolves to `fts5` |

Retrieval defaults to FTS5 (`VOICE_AGENT_RAG_RETRIEVAL` unset means `auto`, which is `fts5` here). If the Pi's SQLite has no FTS5, the retriever falls back to `rank-bm25` by itself; force it with `VOICE_AGENT_RAG_RETRIEVAL=bm25`. The docs MCP server uses `mcp.server.fastmcp` and a lazy import path, so on x86 the docs server imports in about 1.4 s and 56 MB RSS (4.1 s and 106 MB before), and the retrieval-only path in about 41 MB with no numpy, `fastmcp` or MarkItDown loaded. These are x86 figures, not Pi ones.

## Check a Pi

```
uv run python scripts/pi_rag_check.py --folder DIR [--queries FILE] [--k 4] [--in-place] [--compare-bm25] [--tokenizer pretok|porter]
```

| Option | Meaning |
|---|---|
| `--folder DIR` | documents to index (required); not modified unless `--in-place` |
| `--queries FILE` | one query per line, or JSON lines with `q`, `query` or `question`; default is 10 built-in queries |
| `--k N` | passages per query (4) |
| `--in-place` | keep the FTS5 database in `DIR/.rag_cache` instead of a temporary folder (use it to time a real reopen) |
| `--compare-bm25` | also time `rank-bm25` on the same chunks (skipped when not installed) |
| `--tokenizer` | `pretok` (default) or `porter` |

Reading the report:

- **Environment lines**: Python and platform, SQLite version and whether `ENABLE_FTS5` is compiled in, and the FTS5 probe result. If the probe fails the script prints `fallback: rank-bm25` and exits 0; that is the state the agent would be in.
- **Import time and RSS** after importing the retrieval modules, and the list of heavy modules loaded (`numpy`, `fastmcp`, `markitdown`, `rank_bm25`, `voice_agent.rag.hybrid`); `none` is the expected value for the FTS5 path.
- **Cold refresh and FTS5 build**: time to read and chunk the folder, and to fill the index.
- **Reopen to first query**: time to open the stored index and answer one query; compare it with the build time (restart cost). Needs `--in-place` to be meaningful.
- **Query p50 and p95** in milliseconds, **peak RSS** (`VmHWM`) and the database size.
- With `--compare-bm25`: build time, first query, p50, p95 and RSS for `rank-bm25` on the same chunks.

For scale, the x86 benchmark on 5,000 synthetic chunks (`tests/rag/test_fts5_bench.py`) gave FTS5 about 54 MB, reopen 68 ms and p50 9 ms, against `rank-bm25` 127 MB, a 0.5 s rebuild and p50 10 ms. Expect the Pi 4 to be slower; run the script on the real folder to know by how much.

Testing on a real Pi: follow `docs/RAG-PI-HANDOFF.md` (test plan and report format); the measurements so far are in `docs/RAG-EXPERIMENTS.md`.
