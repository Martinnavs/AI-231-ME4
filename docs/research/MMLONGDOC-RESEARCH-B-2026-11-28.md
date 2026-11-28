# MMLongBench-Doc Open-Source Projects & Text-First Pipeline Research (B)

**Date:** 2026-11-28 | **Target:** ME4 post-deploy document lane (Raspberry Pi 4, 4 GB RAM, no GPU/VLM at query time; SQLite FTS5 + Gemini Flash-Lite / Qwen3.8-27B)
**Verification Legend:** `VERIFIED` = directly observed in repo files or primary-source search results read on `2026-11-28`; `UNVERIFIED` = exact sub-figure, commit date, or split could not be confirmed from primary source in this session.

---

## 1. ME4 Baseline & Gap Summary (Verified from Repo)

ME4's production document lane (`pipeline/src/voice_pipeline/ask_documents.py`, `src/voice_agent/rag/dbindex.py`, `src/voice_agent/rag/fts5.py`, `src/voice_agent/rag/docindex.py`) runs **SQLite FTS5** (`pretok`/`porter`, `OR`-joined up to 64 terms, no embeddings) via `DbIndex.file_context(n_files=3, whole_max_chars=15_000, per_big=4, grep_top=20, rank_k=300)`, followed by a single structured JSON LLM call (`quotes`, `calc`, `file`, `answer`) and deterministic checks (`_run_checks`). PDFs are parsed with `pypdfium2` (`get_text_range()`, `<20 chars/page` marked sparse).

On the frozen 25-PDF MMLongBench-Doc dev split (`109` questions: `82` in-scope answerable, `24` unanswerable, `3` `out_of_scope_vision`; `1,039` pages; `23.3%` scanned; `out/mmlongdoc/corpus25/ingest.json`: `3,613` chunks, `33.31s`, `50.8 MB` peak RSS, OCR off):

| Metric | Qwen3.8-27B (`score-dev.md`) | Gemini Flash-Lite (`score-dev-gemini.md`) | Root-Cause Breakdown (`diagnose-dev.md`, Qwen 62 no-answer rows) |
| :--- | :---: | :---: | :--- |
| **Headline (in-scope answerable, n=82)** | **0.129** | **0.1189** | **R0 (not self-contained):** 43/62 (hand-rated: >=27/82 of all dev rows cannot be answered in a 25-doc shared corpus; 16/43 are self-contained real misses) |
| **In-scope incl. unanswerable (n=106)** | 0.3262 | 0.309 | **R1 (gold page not retrieved):** 13/62 (**0% scanned**; **11/13 are Table/Float**) |
| **Official overall incl. vision (n=109)** | 0.3172 | 0.3005 | **R2 (page retrieved, value missing):** 3/62 (all 3 are derived/calculated ratios/deltas, not extraction errors) |
| **Official F1 (unanswerable-aware)** | **0.2095** | **0.1912** | **R3 (gold in passage, lane declined):** 3/62 (all 3 failed `key_absent` on spelling differences) |
| **Unanswerable accuracy (n=24)** | **1.000** | 0.9583 | **Accepted (n=20):** 8 full (1.0), 3 partial, 9 scored 0; `lane:none=76`, `check_failed=13` (`key_absent` 8/13, `names_numbers` 3/13, `calc` 2/13) |
| **By evidence: Text / Table / Layout** | 0.1978 / 0.0833 / 0.0652 | 0.20 / 0.0833 / 0.0 | **Zero-chunk slide decks:** 2 image-only PDFs yielded 0 chunks (5 dev rows); **P6:** `FY2021` vs `fiscal 2021`, `Bestbuy` vs `Best Buy`, `Rupee` vs symbol |

**Core Diagnosis:** ME4 does not fail mainly on answer generation or unanswerable abstention—it excels at abstaining (`1.0` Qwen, `0.9583` Gemini). It fails on: **(1)** shared-corpus non-self-contained questions (`>=33%` ceiling); **(2)** retrieval & table structure (`Table` `0.0833`, `Float` `0.069`, `11/13` R1 misses are Table/Float because `per_big=4` + `OR`-BM25 + flattened `pdfium` text miss borderless 10-K tables); **(3)** zero-chunk image-only slide PDFs; **(4)** uncomputed multi-step financial ratios (`R2` + self-contained `R0`); and **(5)** overly brittle `key_absent` string matching (`8/13` `check_failed`).

---

## 2. Open-Source Projects & Pipelines on MMLongBench-Doc & Related Benchmarks

### 2.1 Master Comparison Table

*Note: Official MMLongBench-Doc scores below are on the **single-PDF-per-question** task (`1,082` or `1,091` rows) with GPT-4o / VLM extractors, so absolute numbers are NOT directly comparable to ME4's 25-PDF shared-corpus Qwen-extracted baseline (`docs/MMLONGDOC-BENCH.md`).*

| # | Project / Repo | Licence | Maintenance / Activity | Claimed Score & Split | Query-Time Needs | Ingest Offline -> Text-Only to Pi? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **SimpleDoc** (`ag2ai/SimpleDoc`) | Apache-2.0 (`VERIFIED`) | EMNLP 2025 paper/release; exact last commit date `UNVERIFIED` | MMLongBench-Doc full: **50.42%** ACC (Qwen2.5-VL-7B+Qwen3-8B; UNA **62.11%**), **59.55%** (Qwen2.5-VL-32B+Qwen3-30B-A3B; UNA **67.40%**) (`VERIFIED`, ACL Anthology Table 3) | VLM embeddings (ColQwen-2.5) + VLM reasoner | **Partially** — page summaries + dual-cue index can run offline -> text, but native VLM working-memory reader cannot run on Pi 4 |
| 2 | **MDocAgent** (`aiming-lab/MDocAgent`) | MIT (`VERIFIED`) | Last major activity ~Mar 2025 (`VERIFIED`) | MMLongBench-Doc: **0.269** (top-1) / **0.347** (top-4) in MDocAgent paper (`VERIFIED`); **43.80%** (7B+8B) / **55.30%** (32B) in SimpleDoc Table 3 (`VERIFIED`); LongDocURL **0.323** top-4 (`VERIFIED`) | ColBERTv2 + ColPali + 5 LLM/VLM agent calls | **No (as-is)** — 5 multi-modal agents too slow/heavy; text-agent + critical verification pattern is adaptable |
| 3 | **ViDoRAG** (`Alibaba-NLP/ViDoRAG`) | Apache-2.0 (`VERIFIED`; underlying Qwen-VL weights carry model terms) | EMNLP 2025 release (`VERIFIED`); exact last commit `UNVERIFIED` | ViDoSeek: **72.1–72.8%** end-to-end ACC (`VERIFIED`, >10% over baselines); also evaluated on SlideVQA-Refined & MMLongBench-Doc (`VERIFIED`, exact MMLongBench number `UNVERIFIED`) | Hybrid visual+text GMM retrieval + VLM Seeker/Inspector/Answerer | **No** — requires page images & VLM inference at query time |
| 4 | **M3DocRAG** (`bloomberg/m3docrag`) | Apache-2.0 (`VERIFIED`) | Paper Nov 2024 (`arXiv:2411.04952`); exact last commit `UNVERIFIED` | MMLongBench-Doc closed-domain top-4 (`ColPali + Qwen2-VL-7B`): **21.0%** ACC, **22.6%** F1 (`VERIFIED`, Table 2) | GPU for ColPali MaxSim + Qwen2-VL-7B | **No** — core premise is pixel-level ColPali retrieval + VLM reading (no OCR/text) |
| 5 | **VisRAG** (`OpenBMB/VisRAG`) | Code Apache-2.0; weights MiniCPM Model License (`VERIFIED`) | 2024–2025 (VisRAG 2.0 follow-up; exact commit `UNVERIFIED`) | Outperforms TextRAG on DocVQA, ChartQA, InfoVQA, SlideVQA (`VERIFIED`, exact SlideVQA number `UNVERIFIED`) | MiniCPM-V visual retriever + VLM generator | **No** — parsing-free vision pipeline |
| 6 | **PageIndex / Mafin 2.5** (`VectifyAI/PageIndex`, `VectifyAI/Mafin2.5-FinanceBench`) | MIT (`VERIFIED`) | Active 2025–2026 (`VectifyAI/MMLongBench-Doc-V2` Apache-2.0) (`VERIFIED`) | **98.7%** on FinanceBench (`Mafin 2.5`, 150 Qs) (`VERIFIED`); MMLongBench-Doc V2 uses semantics-aware LLM judge (`VERIFIED`) | LLM-only (no embeddings, no GPU) traversing hierarchical TOC JSON | **Yes (best fit)** — TOC tree + node summaries built offline on laptop; Pi does tree lookup + LLM call (cost: multi-step LLM calls/latency) |
| 7 | **Docling** (`docling-project/docling`) | MIT (`VERIFIED`) | Actively maintained (`VERIFIED`, 2024–2026) | High table structure accuracy via TableFormer; x86 CPU **~3.1s/page** (~1.57 p/s), Apple M3 Max **~1.27s/page** (`VERIFIED`, Docling Tech Report) | **None at query time** if run at ingest | **Yes (ideal ingest)** — 1.4–2 GB deps + 0.5–1 GB weights too heavy for Pi 4 runtime, ideal on laptop -> clean Markdown to Pi |
| 8 | **MinerU** (`opendatalab/MinerU`) | AGPL-3.0 / custom commercial revenue & MAU clauses (`VERIFIED`) | Very actively maintained (`VERIFIED`) | OmniDocBench table TEDS **93–95%+** (`MinerU2.5-Pro`, `VERIFIED`); 0.4–6.0s/page (`VERIFIED`) | **None at query time** if run at ingest | **Yes (ingest only)** — GPU recommended at ingest; AGPL/commercial license caveat |
| 9 | **Marker** (`VikParuchuri/marker`) | Code GPL-3.0 / Apache-2.0 (`UNVERIFIED` split across versions) + OpenRAIL-M model weights (`VERIFIED`) | Actively maintained (`VERIFIED`) | ~1–3s/page; lower raw table TEDS than MinerU unless LLM pass enabled (`VERIFIED`) | **None at query time** if run at ingest | **Yes (ingest only)** — GPU needed for speed; OpenRAIL-M commercial limits |
| 10 | **PyMuPDF4LLM** (`pymupdf/PyMuPDF4LLM`) | AGPL-3.0 / Commercial (`VERIFIED`) | Actively maintained (`VERIFIED`) | **0.01–0.09s/page** (~60–120 p/s), ~262–300 MB footprint (`VERIFIED`); poor on borderless 10-K tables & multi-col (`VERIFIED`) | Runs on Pi 4 CPU (C MuPDF) | **Yes (runs on Pi 4 too)** — drop-in fast text-to-MD, but no OCR and weak on borderless tables |
| 11 | **RAGFlow / DeepDoc** (`infiniflow/ragflow`) | Apache-2.0 (`VERIFIED`) | Actively maintained (`VERIFIED`) | Modular layout/OCR/table + hybrid BM25/vector; wraps Docling/MinerU/PaddleOCR; no single fixed FinanceBench score (`VERIFIED`) | Needs ES/Infinity + embeddings in default stack | **Partially** — DeepDoc layout/table chunker can export text, but full RAGFlow server too heavy for Pi 4 |
| Ref | **MMLongBench-Doc Baseline** (`mayubo2333/MMLongBench-Doc`) | Apache-2.0 (research note) (`VERIFIED`) | NeurIPS 2024 (`1,082` Qs) / HF `1,091` Qs (`VERIFIED`) | GPT-4o Vision **42.7–44.9%** F1 (+10.8–14.4% over GPT-4o OCR text-only); GPT-4V **30.5–31.4%** F1; human ~66% (`VERIFIED`) | N/A (eval harness) | Reference baseline |

### 2.2 Stage-by-Stage Breakdown of Top 10 Open-Source Systems

1. **SimpleDoc (`ag2ai/SimpleDoc`):**
   * *Stages:* Offline per-page rendering + ColQwen-2.5 visual embeddings + LLM-generated per-page summaries -> Online Stage 1 visual candidate retrieval -> Stage 2 summary-driven filtering/reranking -> Single VLM reasoner with iterative working memory that requests additional pages until confident or abstains.
   * *Smallest driver of score:* **Dual-cue filtering (page summary + visual cue) + iterative multi-turn page fetch**, which cuts retrieved pages to ~3.5/query while jumping `UNA` accuracy to `62.11%`/`67.40%` (`VERIFIED`).
   * *Pi transferability:* The **offline page summary + TOC cue** idea transfers directly to ME4 as text-only enriched chunks without any VLM on the Pi.

2. **PageIndex / Mafin 2.5 (`VectifyAI/PageIndex`):**
   * *Stages:* Parse PDF -> Build hierarchical JSON tree (TOC + page ranges + LLM node summaries, zero vector DB/chunking) -> Agentic LLM tree search over TOC nodes -> Load full text of selected pages -> Answer & verify.
   * *Smallest driver of score:* **Preserving document hierarchy + full-page loading** instead of top-k disconnected chunks (beats 30–50% vector RAG to **98.7%** on FinanceBench, `VERIFIED`).
   * *Pi transferability:* **100% text-only at query time.** Offline laptop builds the compact JSON TOC tree; Pi does FTS5/BM25 file selection + tree navigation or parent-page expansion. *Caveat:* multiple sequential LLM calls per query hurt Pi voice TTFA latency unless pruned to 1–2 calls.

3. **MDocAgent (`aiming-lab/MDocAgent`):**
   * *Stages:* Dual parse (text via OCR/PDF + page images) -> Dual retrieve (ColBERTv2 text + ColPali image) -> General Agent draft -> Critical Agent bottleneck check -> Text Agent + Image Agent parallel deep read -> Summarizing Agent synthesis.
   * *Smallest driver of score:* **Critical Agent feedback loop** that forces re-reading specific gaps before answering (`+12.1%` over single-modal baselines, `VERIFIED`).
   * *Pi transferability:* 5 LLM/VLM calls fail ME4's ~2.5–6s budget; however, a **1-step critique/query-rewrite retry** on `none`/`check_failed` is feasible within ME4's explicit-docs `6.0s` budget (`DOC_TURN_TIMEOUT_S`).

4. **ViDoRAG (`Alibaba-NLP/ViDoRAG`):**
   * *Stages:* Page image + text indexing -> **GMM-based dynamic hybrid retrieval** (selects adaptive `k` per query from score distribution instead of fixed top-k) -> Seeker -> Inspector -> Answerer iterative agents.
   * *Smallest driver of score:* **Dynamic `k` selection via score distribution (GMM)** + Inspector filtering noisy pages (`VERIFIED`).
   * *Pi transferability:* Dynamic `k` / per-document page-budgeting is pure math on BM25 scores and costs <1 ms on Pi 4.

5. **M3DocRAG (`bloomberg/m3docrag`):**
   * *Stages:* Render PDF pages -> ColPali late-interaction patch embeddings -> MaxSim top-k pages -> Qwen2-VL-7B reader.
   * *Smallest driver of score:* Skipping lossy OCR so tables/charts stay intact (`21.0%` ACC / `22.6%` F1 vs text-only ColBERTv2+Llama, `VERIFIED`).
   * *Pi transferability:* Not usable on Pi 4 (`ColPali` + 7B VLM), except as proof that **if a page has images/slides/charts, vision must run offline at ingest and emit text captions/tables**.

6. **VisRAG (`OpenBMB/VisRAG`):**
   * *Stages:* Page-as-image encoding via MiniCPM-V (`VisRAG-Ret`) -> cosine retrieval -> VLM generation.
   * *Smallest driver of score:* Joint vision-language embedding space avoiding OCR drops (`VERIFIED`).
   * *Pi transferability:* Incompatible with text-only Pi 4 runtime.

7. **Docling (`docling-project/docling`):**
   * *Stages:* `pypdfium2` backend -> DocLayNet layout detection -> TableFormer table-structure recognition -> RapidOCR on bitmap regions -> unified `DoclingDocument` -> Markdown/JSON chunking (also natively parses PPTX, DOCX, XLSX, HTML).
   * *Smallest driver of score:* **TableFormer + layout reading-order model**, preserving borderless financial tables and multi-column flow (`VERIFIED`).
   * *Pi transferability:* Run **offline on laptop at ingest** -> export structured Markdown (`#` headings, GitHub pipe tables, slide text) -> ship plain Markdown/SQLite DB to Pi 4.

8. **MinerU (`opendatalab/MinerU`):**
   * *Stages:* Layout segmentation -> formula/table VLM (`MinerU2.5`) -> OCR -> reading-order sort -> Markdown/JSON.
   * *Smallest driver of score:* VLM table/formula parsing (`93–95%+` TEDS on OmniDocBench, `VERIFIED`).
   * *Pi transferability:* Laptop/GPU offline ingest only; license (`AGPL-3.0` + commercial MAU/revenue clauses) requires legal care.

9. **Marker (`VikParuchuri/marker`):**
   * *Stages:* Text extraction + Surya layout/OCR -> Markdown conversion -> optional LLM table/equation repair pass.
   * *Smallest driver of score:* Surya reading-order + optional LLM table cleanup (`VERIFIED`).
   * *Pi transferability:* Laptop offline ingest only; OpenRAIL-M restricts commercial scale.

10. **PyMuPDF4LLM (`pymupdf/PyMuPDF4LLM`):**
    * *Stages:* C-level MuPDF span/font inspection -> heuristic header detection + basic `find_tables()` -> page-chunked Markdown.
    * *Smallest driver of score:* Font-size heading hierarchy at C speed (`0.01–0.09s/page`, `VERIFIED`).
    * *Pi transferability:* Runs directly on Pi 4 CPU, **but** lacks built-in OCR and scrambles borderless 10-K tables (`VERIFIED`).

11. **RAGFlow / DeepDoc (`infiniflow/ragflow`):**
    * *Stages:* DeepDoc layout + OCR + table recognition (or pluggable Docling/MinerU adapter) -> template-based chunking (paper/laws/manual/table) -> hybrid dense + BM25 retrieval + rerank.
    * *Smallest driver of score:* Document-type-specific chunking templates (e.g., keeping tables atomic as whole chunks instead of arbitrary 512-token splits, `VERIFIED`).
    * *Pi transferability:* Atomic table/slide chunking rules transfer directly to `docindex.py` with zero runtime overhead.

---

## 3. Deep-Dive 1: PDF-to-Text/Markdown Parsers (Tables, Layout, Slides, Pi 4 Feasibility)

ME4 currently uses `pypdfium2` `get_text_range()` (`src/voice_agent/rag/docindex.py:1476-1520`), which dumps raw character streams per page without layout, reading-order, or table grid reconstruction, and skips pages `<20 chars` (causing 2 slide-deck PDFs in our 25-doc corpus to emit **zero chunks**).

| Parser | License | Speed (x86 / Pi 4) | RAM / Disk | Borderless 10-K Tables | Multi-Column Layout | Image-Only Slide PDFs | Pi 4 Query/Ingest Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`pypdfium2` (ME4 current)** | Apache-2.0 / BSD (`VERIFIED`) | ~0.03s/page (25 PDFs / 1,039 pp in `33.3s` on laptop, `VERIFIED`) | **50.8 MB** peak RSS (`VERIFIED`) | **Poor** (flattens rows/cols into stream) | **Poor** (interleaves columns) | **0 chunks** unless `--ocr` fixed | Keep as fast Tier-1 digital reader on Pi |
| **`PyMuPDF4LLM`** | AGPL-3.0 / Commercial (`VERIFIED`) | **0.01–0.09s/p** (`VERIFIED`); ~10–15s / 60 pp | ~200–400 MB RAM; ~262–300 MB install (`VERIFIED`) | **Poor–Mod** (`find_tables()` needs grid lines; fails borderless 10-Ks, `VERIFIED`) | **Poor–Mod** | **0 chunks** (no built-in OCR, `VERIFIED`) | Fast on Pi 4, but AGPL license + weak on 10-Ks |
| **`pdfplumber`** | MIT (`VERIFIED`) | **Slow** (~3–5s+ per 10-K, pure Python `pdfminer.six`, `VERIFIED`) | Low–Mod (~200–500 MB) | **High on digital PDFs** via coordinate/whitespace tuning (`VERIFIED`); fails multi-page stitching | **Mod** (with bbox rules) | **0 chunks** (no OCR) | **Best MIT lightweight fallback** on Pi/laptop for table-heavy digital PDFs |
| **`Docling` (IBM)** | **MIT** (`VERIFIED`) | **~3.1s/p** x86 CPU; **1.27s/p** M3 Max (`VERIFIED`); much slower on Pi 4 | **1.4–2 GB** deps + **0.5–1 GB** weights; ~1–2 GB RAM (`VERIFIED`) | **Very High** (`TableFormer` handles borderless 10-K indents & grids, `VERIFIED`) | **Excellent** (`DocLayNet` reading order) | **Yes** (built-in **RapidOCR** / ONNX fallback) | **Avoid at Pi runtime**; **#1 choice for laptop offline ingest** -> Markdown to Pi |
| **`MinerU`** | AGPL-3.0 + commercial clauses (`VERIFIED`) | 0.4–6.0s/p GPU/CPU (`VERIFIED`) | High (VLM weights, `VERIFIED`) | **Highest** (93–95%+ TEDS OmniDocBench, `VERIFIED`) | **Excellent** | **Yes** (OCR + VLM) | **Avoid on Pi**; laptop GPU only; license risk |
| **`Marker`** | Code GPL-3.0 + OpenRAIL-M (`VERIFIED`) | 1–3s/p (`VERIFIED`) | High VRAM/RAM | **Mod–High** (needs LLM pass, `VERIFIED`) | **Very Good** | **Yes** (Surya OCR) | **Avoid on Pi**; OpenRAIL-M limits |
| **`olmOCR` / `Nougat`** | Apache-2.0 / MIT (`VERIFIED`) | GPU-bound (`allenai/olmocr` needs modern NVIDIA GPU + SGLang/vLLM, `VERIFIED`) | High VRAM | High (olmOCR) | High | Yes (`olmOCR`) | **Avoid on Pi**; laptop/cloud GPU only |

**Solving the Zero-Chunk Slide Decks & Broken `--ocr`:**
1. As documented in `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md`, ME4 already has **RapidOCR** (`src/voice_agent/rag/ocr.py`), but `.zip --ocr` fails due to `work_dir` deletion (`ingest.py:660,781`), `images.py:147-149` has a `px^2` vs `pt^2` `17x` mismatch, and `ocr.py:360` (`filter_lines`) is a no-op. Fixing T5 immediately recovers text from image-only slide PDFs without adding new ML dependencies!
2. For **offline laptop ingest -> text shipped to Pi**: running **Docling** (MIT, CPU-friendly) or **hybrid routing** (`pypdfium2` for plain prose pages; `pdfplumber` or `Docling` for pages with dense numeric alignments / tables; and **RapidOCR** or a **single offline VLM/Gemini Flash-Lite captioning pass** on `<20 chars/page` slide pages) yields clean Markdown tables + slide transcripts stored directly in `corpus.db` as plain text.

---

## 4. Deep-Dive 2: Numeric QA Over Financial Tables (Code Computation, Not LLM Guessing)

ME4 **already has** a safe AST calculator `_eval_calc` (`pipeline/src/voice_pipeline/ask_documents.py:226-286`)! When the LLM emits `"calc": "10314 / 7674"`, `_run_checks` verifies every operand appears in `file_passages` and `_norm_num(str(calc_val))` appears in `answer`. Yet **Float accuracy is 0.069 (`2/29`)**, **Table is 0.0833**, **11 of 13 R1 misses are Table/Float**, and **all 3 R2 misses** (`q1018` Costco debt/EBITDA, `q1032` Netflix personnel-cost increase, `q1043` Nike payables turnover) failed because the computed number isn't literally in the text!

What successful open-source systems (**FinQA/PoT**, **DocFinQA**, **TableRAG**, **Chain-of-Table**, **PageIndex/Mafin**) show:
1. **Why PoT / Code Execution beats LLM arithmetic:** On **FinQA**, offloading arithmetic to Python (`Program-of-Thoughts` / `PAL`) reaches **~69–74%** execution accuracy with GPT-4 (`VERIFIED`, Sec. 4 `arXiv` evaluations), removing mental-math errors.
2. **Why ME4's `_eval_calc` fails in practice (3 bottlenecks):**
   - **Bottleneck A — Retrieval misses the second table (`11/13` R1):** Ratios like `inventory turnover` (`COGS / avg inventory`), `total debt / EBITDA`, or `payables turnover` require **two different financial statements** (Income Statement + Balance Sheet + Cash Flow / Debt footnote, often 10–30 pages apart). ME4's `per_big=4` returns at most 4 chunks from that 10-K, frequently fetching MD&A commentary instead of both financial tables!
   - **Bottleneck B — Flattened `pdfium` tables break column/unit association:** Without Markdown pipe tables (`| Line Item | FY2021 | FY2020 |`) and preserved table captions (`"in millions"`), the LLM cannot reliably map `"Cost of sales"` to the right year column or scale units, so it emits `none` (`76/109`) rather than guessing a `calc` expression.
   - **Bottleneck C — Strict `_eval_calc` operand/rounding guard:** `_run_checks` (`ask_documents.py:942-956`) rejects `calc` if the LLM includes intermediate rounding, unit conversion (`* 1e6` when `1e6` isn't literally in `pnums`), or a ratio formatted `0.97` vs `0.967` (`calc` failed `2/13` `check_failed` rows).
3. **Practical Text-First Fix (TableRAG / Chain-of-Table / FinQA pattern):**
   - Extract tables atomically as **whole Markdown tables** (via `Docling` at offline ingest, or `pdfplumber`/coordinate grouping for digital PDFs) with **table caption + unit header (`in millions`) + section heading** prepended to every table chunk.
   - Expand `_eval_calc` safely to allow standard financial unit constants (`100`, `1000`, `1e6`, `365`, `2` for averages like `(beg + end) / 2`) without requiring `2` or `365` to appear verbatim in `pnums`, and check tolerance (`abs(ans - calc_val) <= 0.02` or 1% relative tolerance) rather than exact substring match.
   - When a query asks for a **financial ratio or multi-period change** (`turnover`, `ratio`, `margin`, `cycle`, `increase`, `percent`), retrieve **dedicated table chunks** (`chunk_type == 'table'`) + **adjacent Balance Sheet / Income Statement pages** (PageIndex / RAGFlow atomic-table pattern).

---

## 5. Deep-Dive 3: Embedding-Free Sparse/BM25 Retrieval & CPU Reranking Enhancements

ME4 relies purely on SQLite FTS5 (`OR`-joined up to 64 tokens, `bm25(weights)` on body + header) with no embeddings. Research and ME4's own prior experiment (`scripts/experiments/rerank_proto.py` E9b) establish clear rules on what works vs what fails on CPU/Pi 4:

1. **Contextual Chunk Prefixes (Anthropic Contextual Retrieval):**
   - Anthropic (`2024`, `VERIFIED`) showed that prepending a short 50–100 token document/section context to each chunk before indexing cuts top-20 retrieval failure rates by **35%** (contextual embeddings), **49%** (contextual embeddings + contextual BM25), and **67%** (+ reranking).
   - *Pi Cost:* **Zero at query time!** In ME4, deterministic contextual prefixes (without even needing an LLM call, or using 1 cheap offline pass at ingest) can prepend `[Doc: <title/company/year> | Section: <heading> | Page: <N> | Type: <table/slide/text>]` directly into the indexed FTS5 `header`/`body` columns. Today `file_cards.py` already extracts `title`, `sections`, `entities`, and `dates`, yet `dbindex.py:424-478` does **not** inject these into FTS5 chunk prefixes or retrieval!

2. **Page-Level & Parent-Window Expansion (SimpleDoc / PageIndex / ViDoRAG pattern):**
   - Chunk-only `per_big=4` splits multi-paragraph answers and table footnotes across chunk boundaries. SimpleDoc averages ~3.5 *full pages*; PageIndex loads *full pages* of matched TOC nodes.
   - *Pi Cost:* Zero ML overhead—when FTS5 hits a chunk on page `P` of a big PDF, return **page `P` (merged chunks of page `P`) + `P±1` table/caption context** or bump `per_big` from `4` to `8–10` (or dynamic `k` via score gap, ViDoRAG style) within Gemini Flash-Lite's context window (still <30 KB text, <$0.001/call).

3. **Query Expansion / Multi-Query Lexical Reformulation (MDocAgent / BM25 PRF):**
   - In `file_context`, `"FY2021"` only matches `"FY2021"`, missing `"fiscal 2021"` or `"2021"`; `"Bestbuy"` misses `"Best Buy"`; `"inventory turnover"` misses `"cost of sales"` + `"inventories"`.
   - *Zero-embedding fixes on Pi:* (a) Deterministic synonym/format expansion in `_build_match_query` (`FY2021` -> `"FY2021" OR ("fiscal" "2021")`, camelCase/compound split `Bestbuy` -> `"Best Buy"`, financial ratio concept dictionary `inventory turnover -> cost of sales merchandise inventories`); (b) **BM25 Pseudo-Relevance Feedback (RM3)** or a **1-shot fast query-expansion fallback** when `ask_documents` returns `none` on explicit/in-scope queries.

4. **Ultra-Lightweight CPU Rerankers (FlashRank / ONNX Cross-Encoders) vs Naive Lexical Proximity:**
   - **Do NOT use naive lexical proximity reranking:** ME4's own `scripts/experiments/rerank_proto.py` (E9b) already proved that window-proximity lexical reranking *dropped* `hits@4` from `92` to `81–87` (`112` rows, `VERIFIED`).
   - **FlashRank (`PrithivirajDamodaran/FlashRank`, Apache-2.0, `VERIFIED`):** Pure `onnxruntime` CPU cross-encoder (no PyTorch/Transformers). Models: `ms-marco-TinyBERT-L-2-v2` (**~4 MB**), `ms-marco-MiniLM-L-12-v2` (**~23–34 MB** INT8/FP32), **<150 MB RAM**, ARM64 compatible, **~0.1–0.4s** on desktop CPU for 100 docs (`VERIFIED`; exact Pi 4 latency `UNVERIFIED`, estimated ~0.3–1.2s for top-20–30 passages). Reranking FTS5 top-30 passages down to top-8 before sending to Gemini gives semantic passage ordering *without* vector embeddings!
   - **BM25S (`xhluca/bm25s`, MIT, `VERIFIED`):** Useful only if replacing SQLite FTS5, which is unnecessary since SQLite FTS5 already executes in <10 ms on 3,613 chunks.

---

## 6. Deep-Dive 4: Abstention on Unanswerable Questions

On MMLongBench-Doc (`~20.6%` unanswerable), models frequently hallucinate instead of abstaining:
- **SimpleDoc** achieves **62.11%** (`7B`) and **67.40%** (`32B`) on `UNA` (`VERIFIED`) vs **MDocAgent** `33.49%` / `45.41%` (`VERIFIED`). SimpleDoc's lever is **working-memory sufficiency gating**: its iterative loop checks whether the retrieved pages *actually contain* the evidence; if exhausted without finding supporting facts, it explicitly outputs `"Not answerable"`.
- **MMLongBench-Doc-V2** (`VectifyAI/MMLongBench-Doc-V2`, `VERIFIED`) moves to a semantics-aware judge that penalizes confident guesses on unanswerables.
- **ME4's Current Abstention Profile:** ME4 is **already world-class at precision/abstention** (`1.000` on `n=24` unanswerables with Qwen, `0.9583` with Gemini), driven by its two-layer guard: prompt-level `"NONE"` + deterministic `_run_checks` (`quote_spans`, `names_numbers_ok`, `_key_absent`, `overlap_ok`).
- **The Pathology — Over-Abstention (`8/13` `check_failed` from `_key_absent`):** As `POTENTIAL-IMPROVEMENTS-PARKED.md` (P6) and `diagnose-dev.md` show, `_key_tokens` (`ask_documents.py:691`) extracts capitalized mid-question words (`FY2021`, `Bestbuy`, `Erstwhile`, `Hotels`, `Limited`, `Rupee`) and `_key_absent` kills **verbatim-correct** answers (`q0939`, `q1025`, `q1083`) whenever spelling/currency notation differs! Meanwhile, `quote_spans` + `names_numbers_ok` + `_narrow_to_named` already block wrong-document hallucinations.

---

## 7. Ranked Concrete Changes for ME4 (Post-Deploy Branch)

Ranked by **Expected Value / (Effort × Pi Cost × Risk)**. Every change preserves text-only Pi 4 runtime and is directly measurable on the frozen 25-doc MMLongBench-Doc dev split (`109` rows, **never tune on `test`**) plus the 100-row sufficiency set.

| Rank | Concrete Change | Expected Value / Effort / Pi Cost / Risk | Implementation Sketch (Target Files) | Exact Measurement with Existing Harness |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Fix/Replace `_key_absent` normalization & unit-token filter (P6)** | **High / XS / Zero Pi cost / Low** | In `ask_documents.py:691-756` (`_key_tokens`, `_key_absent`): fold `FY<yyyy>` <-> `fiscal <yyyy>`, strip spaces/case for compound brand names (`Bestbuy` <-> `Best Buy`), drop currency/unit words (`Rupee`, `Dollar`, `millions`), and test relaxing `_key_absent` vs keeping it only on file selection. | `cd pipeline && uv run python ../scripts/bench_mmlongdoc.py run --split dev --tag dev-p6 && uv run python ../scripts/bench_mmlongdoc.py score --tag dev-p6`. Check: R3 (`3/3` recovered?), `check_failed` drop from `13`, `unanswerable` acc stays `>=0.95`, plus no regression on 100-row sufficiency controls. |
| **2** | **Offline Docling ingest (laptop) + table-atomic chunking & contextual prefixes** | **Very High / M / Zero query Pi cost / Low** | Add optional offline `docling` converter path (MIT) exporting Markdown (`#` headings, GitHub pipe tables, slide text) -> `corpus.db`. Keep tables atomic (never split mid-table) and prepend `[Doc \| Company/Year \| Section \| Page]` to chunk `header`/`text`. | Run offline Docling ingest on the 25 PDFs -> `bench_mmlongdoc.py run/score --split dev --tag dev-docling` + `bench_mmlongdoc_diagnose.py`. Compare `Table` (`0.0833`), `Layout` (`0.0652`), `Float` (`0.069`), and `R1` (`13`). |
| **3** | **Fix `--ocr` on `.zip`, `images.py` `17x` check & `ocr.py` `filter_lines` (T5)** | **Med-High / S / Ingest-only Pi cost / Low** | Fix `ingest.py:660,781`, `images.py:147-149` (`px^2` vs `pt^2`), and `ocr.py:360` (`filter_lines`) so image-only slide decks & 23.3% scanned pages yield text chunks. | Rebuild `corpus25` with `--ocr` -> verify `ingest.json` chunks >0 for the 2 image-only slide PDFs -> `bench_mmlongdoc.py run/score --split dev --tag dev-ocr`. Measure recovery on the 5 slide-deck dev rows & scanned pages. |
| **4** | **Page-window / dynamic-`k` retrieval + financial ratio query expansion** | **High / S / <5 ms Pi cost / Low** | In `dbindex.py:385-481` (`file_context`): bump `per_big` `4 -> 8` (or merge same-page chunks & include adjacent table page), add deterministic `FY`/compound/ratio-term synonyms (`turnover`, `EBITDA`, `margin`) to `_build_match_query` (`fts5.py:78`). | Run `bench_mmlongdoc.py run/score --split dev --tag dev-ret` + `bench_mmlongdoc_diagnose.py`. Target: drop `R1` (`13`, especially the `11/13` Table/Float misses) without hurting `cited_file_match` (`33/33`). |
| **5** | **Relax `_eval_calc` constants/tolerance + prompt for 2-step table formulas** | **Med-High / S / <1 ms Pi cost / Low-Med** | In `ask_documents.py:226-286,938-960`: allow standard financial constants (`100`, `1000`, `1e6`, `365`, `2`) in `calc` without requiring them in `pnums`, allow ±1% / rounding tolerance against `answer`, and prompt `quotes` across two table passages + `calc` expression. | Run `bench_mmlongdoc.py run/score --split dev --tag dev-calc`. Target: lift `Float` (`0.069`, `n=29`) and convert the 3 `R2` + self-contained `R0` ratio rows while keeping `calc` hallucinations `0`. |
| **6** | **1-retry Critique/Reformulate fallback on `none`/`check_failed` (MDocAgent-lite)** | **Med / M / +1 LLM call on misses / Med** | When `ask_documents` gets `none` or `check_failed` under `DOC_TURN_TIMEOUT_S=6.0`, run 1 fast query-rewrite (e.g., extract entity + financial line items) -> re-query `file_context` -> second verdict. Skip under `2.5s` implicit race. | Run `bench_mmlongdoc.py run/score --split dev --tag dev-retry` & check latency distribution + `unanswerable` false-positive rate (`>=0.95` gate). |
| **7** | **FlashRank ONNX CPU reranker (top-30 -> top-8) on Pi 4** | **Med / M / ~100–150 MB RAM, ~0.3–1s CPU / Med** | Prototype `FlashRank` (`ms-marco-MiniLM-L-6/12-v2` or `TinyBERT-L-2-v2`, Apache-2.0, pure `onnxruntime`) in `dbindex.py` to rerank top-30 FTS5 passages before `per_big` selection. | First measure `hits@4`/`R1` offline on `bench_mmlongdoc_diagnose.py` + `scripts/experiments/` golden set; then benchmark Pi 4 RSS & latency via `pi_health.py` before enabling. |
| **8** | **Benchmark single-doc vs shared-corpus split cleanly in harness** | **High (eval clarity) / XS / Zero / Zero** | Add `--per-doc` or report `self_contained_only` (`n<=55`) alongside shared-corpus (`n=82`) in `bench_mmlongdoc.py` so improvements aren't masked by the `>=27/82` `R0` shared-corpus ceiling. | Run `bench_mmlongdoc.py score` reporting both `shared_corpus` and `single_doc_scoped` on `dev` to separate retrieval gains from impossible `R0` prompts. |

---

## 8. What to Avoid (Heavy Dependencies, GPU-Only Parts, Anti-Patterns)

1. **Do NOT put `Docling`, `MinerU`, `Marker`, `Unstructured` (`hi_res`), or `ColPali`/`ColQwen` on the Pi 4 at runtime:** Docling alone pulls `1.4–2 GB` dependencies + `0.5–1 GB` weights and `~1–2 GB` RAM (`VERIFIED`), risking OOM kills on a 4 GB Pi running Parakeet/STT/agent_server. Use Docling strictly **offline on a laptop at ingest** -> ship text/SQLite to Pi.
2. **Do NOT adopt `MinerU` (`AGPL-3.0` + commercial MAU/revenue weight clauses) or `Marker` (`OpenRAIL-M` weight limits) or `PyMuPDF4LLM` (`AGPL-3.0`) if commercial/proprietary distribution matters** without legal sign-off (`VERIFIED`); prefer **MIT** (`Docling`, `pdfplumber`, `PageIndex`) or **Apache-2.0** (`FlashRank`, `pypdfium2`).
3. **Do NOT use multi-turn `PageIndex` / 5-agent `MDocAgent` loops on live voice turns:** 4–6 sequential LLM calls violate ME4's `1.3–2.5s` TTFA voice latency budget. Borrow PageIndex's *offline TOC/section summary*, not its 5-hop agent loop.
4. **Do NOT add dense vector embeddings (e.g., BGE-M3 / Qdrant / FAISS) before exhausting contextual FTS5 + table parsing + FlashRank:** embeddings add ingest time, RAM, and model weight complexity while still failing on flattened 10-K tables and uncomputed ratios.
5. **Do NOT use naive lexical proximity reranking (`rerank_proto.py` E9b):** already measured negative (`hits@4` fell `92 -> 81–87`, `VERIFIED`).
6. **Do NOT treat `Chart`/`Figure` vision-only MMLongBench-Doc questions as text-lane targets** unless an offline laptop VLM captioner runs at ingest and ships text captions to the Pi.

---

## 9. Primary Sources Consulted (Read 2026-11-28)

- **Repo Evidence:** `docs/MMLONGDOC-BENCH.md`, `out/mmlongdoc/score-dev.md`, `out/mmlongdoc/score-dev-gemini.md`, `out/mmlongdoc/diagnose-dev.md`, `out/mmlongdoc/corpus25/ingest.json`, `docs/POTENTIAL-IMPROVEMENTS-PARKED.md`, `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md`, `pipeline/src/voice_pipeline/ask_documents.py`, `src/voice_agent/rag/dbindex.py`, `src/voice_agent/rag/fts5.py`, `src/voice_agent/rag/docindex.py`, `scripts/bench_mmlongdoc.py`, `scripts/experiments/rerank_proto.py`.
- **MMLongBench-Doc & V2:** [GitHub `mayubo2333/MMLongBench-Doc`](https://github.com/mayubo2333/MMLongBench-Doc), [arXiv:2407.01523](https://arxiv.org/abs/2407.01523), [GitHub `VectifyAI/MMLongBench-Doc-V2`](https://github.com/VectifyAI/MMLongBench-Doc-V2) (`VERIFIED`).
- **SimpleDoc:** [GitHub `ag2ai/SimpleDoc`](https://github.com/ag2ai/SimpleDoc), ACL Anthology EMNLP 2025 Table 3 (`VERIFIED`).
- **MDocAgent:** [GitHub `aiming-lab/MDocAgent`](https://github.com/aiming-lab/MDocAgent), arXiv (`VERIFIED`).
- **ViDoRAG / VisRAG / M3DocRAG:** [GitHub `Alibaba-NLP/ViDoRAG`](https://github.com/Alibaba-NLP/ViDoRAG), [GitHub `OpenBMB/VisRAG`](https://github.com/OpenBMB/VisRAG), [GitHub `bloomberg/m3docrag`](https://github.com/bloomberg/m3docrag) (`arXiv:2411.04952`) (`VERIFIED`).
- **PageIndex / FinanceBench / Table & Code QA:** [GitHub `VectifyAI/PageIndex`](https://github.com/VectifyAI/PageIndex), [GitHub `patronus-ai/financebench`](https://github.com/patronus-ai/financebench) (`arXiv:2311.11944`), `TableRAG` / `DocFinQA` / `Chain-of-Table` (`arxiv.org`) (`VERIFIED`).
- **Parsers & Sparse/Reranking:** [GitHub `docling-project/docling`](https://github.com/docling-project/docling) & Docling Technical Report (`arxiv.org`), [GitHub `opendatalab/MinerU`](https://github.com/opendatalab/MinerU), [GitHub `VikParuchuri/marker`](https://github.com/VikParuchuri/marker), [GitHub `allenai/olmocr`](https://github.com/allenai/olmocr), [GitHub `facebookresearch/nougat`](https://github.com/facebookresearch/nougat), [GitHub `infiniflow/ragflow`](https://github.com/infiniflow/ragflow), [GitHub `PrithivirajDamodaran/FlashRank`](https://github.com/PrithivirajDamodaran/FlashRank), [GitHub `xhluca/bm25s`](https://github.com/xhluca/bm25s), Anthropic Contextual Retrieval (`anthropic.com`) (`VERIFIED`).
