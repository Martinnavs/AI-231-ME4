# RAG experiment log (2026-11-18 .. 2026-11-20)

Chronological record of every measurement behind the document-RAG component: what was asked, how it was measured, the numbers, and what was decided. Written so a new agent can continue without the original conversation. Companion files: `docs/RAG.md` (architecture and options), `docs/RAG-PI-HANDOFF.md` (what to run on the Raspberry Pi), `.scratch/rag-research/` on the owner's cluster (internal research notes R1, R2, R3a, R3b, R3 merged list; not in the repo).

## 0. Measurement protocol (read this first)

**Test sets.**
- Fixture set (in the repo): `tests/fixtures/rag_corpus` (13 small files incl. a pdf and a docx) with `config/eval/rag_golden.jsonl` (29 answerable rows + 7 unanswerable/off-topic) and `config/eval/rag_golden_office.jsonl` (6 xlsx/pptx rows over `tests/fixtures/rag_office`). Short files: good for regressions, weak for ranking decisions.
- External mixed set (NOT in the repo): `~/rag-testsets/ipo-2026/` on the cluster: 44 real public documents (8 pdf, 8 docx, 8 pptx, 8 xlsx, 6 html, 2 csv, 3 md + 1 txt; 73 MB; US federal works, university documents, ONS/Census data, an RFC; licences in `MANIFEST.json`), 133 golden rows (112 answerable: lexical 24, numeric 20, identifier 12, paraphrase 11, table_cell 29, spreadsheet_cell 16; 14 unanswerable, 7 off-topic) with per-file `difficulty` tags (`multi_column`, `merged_cells`, `wide_table`, `hyphenated_line_breaks`, ...). Anchors are verbatim strings (<150 chars) taken from the author's own extraction (pypdf / python-docx / bs4 / openpyxl).
- The external set was built by an agent (Opus) with a verify script (`tools/verify_golden.py`: sha256, anchor found in the stated file and absent from all others, paraphrase overlap rule); 20 random rows were re-read by hand.

**Metrics.** `python -m voice_agent.rag.metrics`: R@1/3/5 and MRR over answerable rows; `--per-query PATH` dumps one JSON row per question (`hit1/3/4/5`, `first_hit_rank`, `top` 5). Production returns k=4 passages (`search_docs top_k=4`), so `hit4` is the production number.

**Matchers (`--match`).** `exact` (whitespace only), `normalized` (case/punctuation/thousands commas), `tokens` (converter-independent: the anchor's word tokens occur in order inside one chunk within a span cap; header text, punctuation and separators between table cells do not matter). **Always report `normalized` and `tokens` when comparing converters or table renderings.** Discovery (E8): the anchors were built from raw text, so a better table rendering (`Grade: A-; GPA: 3.667` instead of `A- 3.667`) counts as a miss under exact/normalized matching although the right file ranks first.

**Statistics.** n=112 answerable rows: a single row is 0.9 points; only differences of about 10 points are detectable by a paired test. Use `scripts/rag_paired_compare.py A.jsonl B.jsonl --metric hit4` (exact McNemar on the discordant pairs, Wilson intervals; standard library only). Most buckets by difficulty have n <= 5: not decision-grade. Attribute gains by ablation (toggle module constants through a wrapper script), not by reverting code. Timings were taken on a shared 256-core node with load average 3-33: order of magnitude only; hit counts are deterministic.

**Baseline control.** Compare against a detached worktree of the previous commit (`git worktree add --detach ~/AI-231-ME4-base HEAD`, `uv sync --group laptop`); copy only harness files (`metrics.py`) into it if it lacks them. The control reproduced earlier numbers exactly (0.670 / 0.589) before each spike.

**Standard external command** (committed default converters, FTS5):
```
python -m voice_agent.rag.metrics --folder ~/rag-testsets/ipo-2026/raw --golden ~/rag-testsets/ipo-2026/golden.jsonl \
  --retriever fts5 --fts5-tokenizer pretok --converters "pdf=legacy,docx=legacy,pptx=direct,xlsx=direct,html=direct,epub=direct" \
  --group-by type,format --match tokens --report-json OUT.json --per-query OUT.jsonl
```

## E1. Baselines on the clean fixtures (2026-11-18)

Question: is BM25 still the right retrieval, and is hybrid worth it? 29 answerable rows. BM25 R@3 0.897 / MRR 0.871; hybrid (BM25 + answerai-colbert-small-v1 late interaction, RRF k=60) R@3 0.931 / MRR 0.888; paraphrase R@3 0.50 vs 0.667. On a 47-question research set BM25 0.723 vs hybrid 0.830 (+5 rows, near the noise floor of ~0.10). Decision then: keep BM25 as the Pi path, hybrid optional on the laptop (`auto`).

## E2. Canonical Markdown spike (2026-11-19)

Idea: convert every file once into canonical Markdown with page/slide/sheet markers (source of truth for chunks, search and verbatim quotes), using MarkItDown for the heavy formats. Findings:
- MarkItDown for pdf/docx: first (cold) conversion 30x (pdf) and 27x (docx) slower than the legacy loaders (cold ratio 30.4 / 27.0; warm cache 0.85x / 0.97x), so the D5 timing criterion fails by design (committed as strict xfail).
- Retrieval on the external set (exact matching): legacy loaders only R@3/MRR 0.455/0.411; MarkItDown for everything 0.545/0.486 (FTS5); best mix (legacy pdf/docx + MarkItDown pptx/xlsx/html) 0.661/0.572.
- Legacy beat MarkItDown for pdf (0.694 vs 0.528) and docx (0.765 vs 0.471; later shown partly anchor bias, E8). Legacy cannot read pptx/xlsx/html at all (0.000).

## E3. Alternatives to BM25 storage (2026-11-19)

- SurrealDB embedded (3.0.0, `mem://`, FULLTEXT BM25): skip. Uncapped OR-query R@3 0.931 but the default AND query returns nothing for natural questions (R@3 0.138 after token filtering), no stemming, no multi-vector storage for late interaction, a 31 MB native wheel, 4-7 ms per query vs 0.56 ms.
- SQLite FTS5 (stdlib): 41 fixture chunks R@3 0.897 / MRR 0.879 vs rank-bm25 0.897 / 0.871; the one miss of the strict pass rule was a length-normalisation difference. At 5,000 synthetic chunks: FTS5 reopen of a persistent index 68 ms, ~54 MB, p50 9 ms, DB 5 MB vs rank-bm25 rebuild 0.5 s, 127 MB, p50 10 ms. Adopted as the default lexical backend (statistical tie in quality, McNemar p >= 0.5; chosen for memory and restart time). LanceDB natively supports multi-vector MaxSim (aarch64 wheel 159 MB: laptop only); Qdrant local mode unverified.

## E4. Converter bake-off: MarkItDown vs Kreuzberg vs direct readers (2026-11-19/20)

Kreuzberg (xberg, MIT, Rust): 36.6 MB wheel, ~0.2 s per small file but flattens PDF tables, no Markdown page markers, native `libx265` GPL-2.0 bundled; retrieval for everything 0.589/0.531 (pptx 0.333, docx 0.529). Kept on branch `exp/kreuzberg`, not in the tree.
Direct readers (openpyxl read-only, python-pptx, lxml.html; isolated child): committed default. Final run on a quiet node, 112 answerable rows, FTS5, exact matching:

| arm | R@3 / MRR | wall | peak RSS |
|---|---|---|---|
| legacy only | 0.455 / 0.411 | 49 s | 210 MB |
| MarkItDown everything | 0.545 / 0.486 | 248 s | 373 MB |
| previous default (MarkItDown for pptx/xlsx/html) | 0.661 / 0.572 | 130 s | 313 MB |
| **direct readers (committed)** | **0.670 / 0.589** | **64 s** | **221 MB** |

Per-file cold start (isolated child, median of 3): MarkItDown xlsx 2.9 s / pptx 2.7 s / html 2.5 s; Kreuzberg 0.36 / 0.33 / 0.33 s; direct 0.95 / 0.62 / 0.30 s. Quality differences between direct and the old default are 1-2 rows (noise); the gain is speed and memory, plus dropping pandas/magika from the path.

## E5. Research reviews (not experiments): R1, R2, R3a, R3b

R1 (measurement reliability): FTS5 vs rank-bm25 is noise (McNemar p >= 0.5); 27 of 43 difficulty buckets have n <= 5; converter-text bias in anchors; a 30 s conversion timeout once removed 6 rows and explained porter's apparent regression; the 2-chunks-per-file cap made R@3 == R@5 on 62 office/html queries; metrics reported k=1/3/5 but production is k=4. R2 (optimizations) and R3 (internals of MarkItDown, Docling, Marker, trafilatura, pdfminer, pdfplumber, mammoth, unstructured, xberg, liteparse, ...): all techniques clean-room REIMPLEMENT (no code copied, no NOTICE needed); do-not-borrow: Marker model weights (OpenRAIL-M), xberg bundled libx265 (GPL-2.0), PyMuPDF/pymupdf4llm (AGPL), MinerU terms, Docling ML models. Ranked list: `.scratch/rag-research/R3-merged-adoption-list.md`.

## E6. Spike 1: ranking, tokenizer, zip guard (2026-11-20, commit 124376e)

Changes: soft cap 2 / hard cap 4 per file with a score gate (0.75 of the best) and cross-file Jaccard 0.75 dedup in `DocIndex.search`; number-preserving tokenizer ("1,599" == "1599", "12.5", comma groups <= 5); FTS5 text column without header tokens; schema stamp 2; zip ratio limit only for parsed members (media <= 4 MB each and 8 MB combined when high-ratio). Paired against the committed default, exact (`normalized`) matching:

| | before | after |
|---|---|---|
| hits at k=4 | 75 / 112 (0.670) | 83 / 112 (0.741): 8 gained, 0 lost, McNemar p = 0.008 |
| R@3 / R@5 / MRR | 0.670 / 0.696 / 0.589 | 0.723 / 0.759 / 0.626 |

Ablation (cap lift disabled): the cap lift alone is +6 rows at k=4 (0 lost, p = 0.031); the zip fix gives the 2 `ehlr_module1.pptx` rows; tokenizer/header neutral on this set. Cost on the clean fixtures: R@5 0.931 -> 0.897, MRR 0.871 -> 0.862 (short files; floor in `test_canonical_spike.py` lowered with the reason in a comment). The committed passage snapshot was regenerated (52 of 67 questions got different candidates).

## E7. Security static scans (security-auditor agent, each followed by fix tickets and a re-check)

Rounds: MarkItDown path (09a-c), FTS5 backend (S1/F02b), direct readers (K2c/K2d), spike 1 (zip media rule, tokenizer regex), spike 2 (docx table output multiplication: High, fixed; html removal quadratic; keep-rule scope; tokens budget). All closed and re-verified. Accepted/deferred: D2 (quote exists but may not answer), canonical cache unauthenticated, forged-marker lines in legacy md/txt/csv (strict xfail), shared FTS5 DB filename across tokenizers, FTS5 N1/N2 (long waits under a held write lock; non-corruption IO errors rebuild).

## E8. Spike 2a: docx tables/styles, html over-pruning, and the matcher (2026-11-20, commit 616c74a)

Changes: stdlib docx loader (native tables, gridSpan/vMerge/tblHeader, heading levels from styles, tracked deletions, footnotes; output caps); html reader guards (wrapper-form demotion, FAQ keep, main/article ancestor protection, link density, over-pruning rollback); `--match tokens`. Same change, two matchers (paired vs 124376e, 112 rows):

| matcher | hits k=4 before -> after | rows gained / lost | note |
|---|---|---|---|
| `tokens` | 90 -> 92 | 2 / 0 (p = 0.5) | html 9/11 -> 11/11, table_cell R@3 0.793 -> 0.862, docx unchanged |
| `normalized` | 83 -> 80 | 0 / 3 (p = 0.25) | all 3 are docx table rows whose source stays at rank 1 (anchor `A- 3.667` vs row `Grade: A-; GPA: 3.667`) |

Conclusion: the apparent docx regression and most of the html "weakness" were anchor-format bias. Under `tokens` the committed default is R@3 0.786 / MRR 0.679 (not 0.723). Weakest buckets under `tokens` (n >= 4, R@3): broken_words_across_lines 0.25 (n=4), table_layout 0.25 (4), dense_numeric_table 0.40 (5), long_document 0.57 (7), figure_embedded_text 0.60 (5), liquid_template_tags 0.60 (5), multi_column 0.68 (19), wide_table 0.70 (23). The UH calendar stays weak: its content is JavaScript-rendered and empty in the raw HTML.

## E9. PDF diagnosis (2026-11-20; scripts in `scripts/experiments/`)

E9-0. Existing canonical PDF post-processing (header/footer filter, dehyphenation, ligatures) via `pdf=pypdf` vs the default legacy PDF path: 2 PDF rows gained (tab-4, lex-2), 2 lost (id-1, num-9), identical hit counts under both matchers (27/36 pdf rows at k=4, p = 1.0). A wash.

E9a (`deep_rank.py`). Where does the answer chunk rank for the 9 PDF misses at k=4 (tokens matcher)? In 7 of 9 the right FILE is in the top 4 (often rank 1) but the answer-bearing chunk is buried:

| row | raw rank of the answer chunk | chunks in that file |
|---|---|---|
| para-1 | 830 of 836 | 456 |
| tab-4 | 95 of 4330 | 911 |
| num-3 | 94 of 3701 | 911 |
| num-4 | 27 of 2694 | 911 |
| para-3 | 49 of 527 | 5 |
| para-2 | 19 of 883 | 188 |
| num-11 | 12 of 1610 | 5 |
| tab-11 | 6 of 761 | 5 |
| lex-2 | 3 of 681 | 188 |

(13,780 chunks in total.) Conclusion: not an extraction problem; within-document chunk ranking (and, for paraphrases, lexical mismatch).

E9b (`rerank_proto.py`). Second-stage lexical reranker (query-term coverage x smallest window containing them, over the top 100-200 BM25 candidates): NEGATIVE. Hits at k=4 on 112 rows: baseline 92; 87 / 87 / 86 / 81 for four parameter settings (lost 7-13 rows, gained 2-3). Not built.

E9c (`sweep_cap.py`). Cap/gate constants (soft, hard, ratio, jaccard), 14 configurations on one index build: flat plateau. Only lex-2 gains (+1 row) at ratio <= 0.6 or soft 3; ratio 0.9 loses 4; soft 1 or hard 2 lose 4-6. Committed (2, 4, 0.75, 0.75) kept.

## E10. Hybrid retriever vs FTS5 on the external set (2026-11-20)

`--retriever hybrid` (rank-bm25 + ColBERT late interaction via the embeddings sidecar, RRF) vs FTS5, `--match tokens`, 112 rows, paired:

| metric | FTS5 | hybrid | gained / lost | McNemar p |
|---|---|---|---|---|
| hits k=3 | 90 (0.804) | 97 (0.866) | 9 / 2 | 0.065 |
| hits k=4 (production) | 92 (0.821) | 98 (0.875) | 7 / 1 | 0.070 |
| hits k=5 | 94 (0.839) | 101 (0.902) | 7 / 0 | 0.016 |
| MRR | 0.697 | 0.755 | | |

Gained at k=4: lex-2, para-2, para-7, para-10, id-9, ss-3, tab-21 (paraphrase R@3 0.455 vs ~0.18); lost: lex-13. Hits at k=4 by format (FTS5 -> hybrid): pdf 27 -> 29, csv 2 -> 4, md 5 -> 6, xlsx 17 -> 18, docx/html/pptx/txt unchanged. Cost: 17 min 52 s wall for the first build over 13,780 chunks, 1.85 GB peak RSS (vectors are cached afterwards). Laptop only. On the laptop `auto` already resolves to hybrid when `~/.venvs/rag-embed` exists; the Pi has no embeddings environment, so the Pi-vs-laptop gap on this set is about 6 of 112 rows at k=4, mostly paraphrase.

## Open hypotheses and next experiments (ranked)

1. **Pi-sized semantic arm.** A static-embedding model (model2vec / potion family: token embeddings looked up and averaged in numpy, tens of MB, milliseconds per query) fused with FTS5 by RRF. Hypothesis: recovers most of the paraphrase gain of E10 on the Pi. Test: build the arm in a separate thin env, run the standard external command with it, paired against FTS5 and against hybrid (`hit4`, `--match tokens`), plus latency/RSS on the Pi. Acceptance: >= 4 of the 6 rows hybrid gains, memory < 150 MB, query < 50 ms on a Pi 4.
2. **PDF chunk structure** (R3a T2, T7): repeat the table header per chunk and carry font-size-detected headings into chunk headers; targets `dense_numeric_table`, `wide_table`, `long_document`. E9a says answer chunks are present but buried, so measure with `deep_rank.py` before/after.
3. **Parked (low value under the `tokens` matcher):** xlsx `<mergeCells>`/multi-level headers, CSV multiline cells and preambles, EPUB `text/html` items; Pi-native PDF reading-order extraction (~8 days, spike 3 in the R3 list).
4. **Pi measurements** (this handoff): none of the latency, memory and timeout numbers above exist for aarch64.
5. **Measurement hygiene:** extend `--match tokens` reporting to `evaluate()`'s default reports; add converter-independent golden rows; grow strata to n >= 35 per decision bucket (R1).

## What was NOT measured

Anything on a Raspberry Pi; hybrid with an FTS5 lexical arm (the hybrid arm uses rank-bm25, `DEFAULT_HYBRID_LEXICAL`); real user documents (only the public external set); answer quality of the full agent (only retrieval hits); effects of OCR or scanned PDFs (out of scope); multi-session concurrency on one docs folder.
