# MMLongBench-Doc Research A: Literature, Leaderboard, Text-Only Ceiling & ME4 Recommendations

**Date:** 2026-11-28 | **Target:** ME4 post-deploy branch (Raspberry Pi 4, 4 GB RAM, no GPU, text-only query lane)  
**Verification Legend:**
- `VERIFIED (repo)`: Directly observed in this repository's files (`read_file` / `run_command` on 2026-11-28).
- `VERIFIED (repo + prior primary)`: Cross-checked against primary-source facts recorded in `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md`.
- `UNVERIFIED (search_web, 2026-11-28)`: Synthesized from `search_web` grounding summaries citing the linked primary URL; raw external PDF/HTML was not directly fetched (no page-reading tool in this environment and shell networking is disabled).

---

## 1. ME4 Baseline & Repo Diagnosis (Ground Truth)

All numbers below are `VERIFIED (repo)`.

### 1.1 Current Harness & Dev Scores
ME4 evaluates MMLongBench-Doc (`HF yubo2333/MMLongBench-Doc @ 2ff6aa92`, 1,091 rows / 135 PDFs) using a document-first 25-PDF shared corpus (`eval/mmlongdoc/manifest.json`: 1,039 pages total, 242 scanned `<20 chars/page` = 23.3%, 200 sampled Qs: 109 dev / 91 test; `docs/MMLONGDOC-BENCH.md:86-129`). Retrieval runs across all 25 PDFs via `DbIndex.file_context` (`n_files=3, whole_max_chars=15000, per_big=4, grep_top=20, rank_k=300`, `dbindex.py:385-481`), with `scripts/bench_mmlongdoc.py:1176-1188` calling `ask_documents` with `web=None, catalog=None, file_has=None, allow_fallback=False`, then scoring with vendored official rules (`scripts/mmlongdoc_eval_score.py`) after Qwen3.8-27B JSON extraction.

| Metric (`n`) | Qwen3.8-27B (`score-dev.md`) | Gemini (`score-dev-gemini.md/.json`) | Notes |
| :--- | ---: | ---: | :--- |
| **Headline: in-scope answerable** (`82`) | **0.1290** | **0.1189** | Swapping to Gemini lowered accuracy (-1.01 pts) |
| In-scope incl. unanswerable (`106`) | 0.3262 | 0.3090 | Core eval set |
| Official overall incl. vision (`109`) | 0.3172 | 0.3005 | 3 `out_of_scope_vision` rows = 0.0 |
| Official F1 (in-scope, `106`) | **0.2095** | **0.1912** | Overall F1 (`109`) on Gemini = 0.1857 |
| Unanswerable (`24`) | **1.0000** | 0.9583 | Abstention is NOT the bottleneck |
| By evidence: Pure-text (`40`) | 0.1978 | 0.2000 | Best evidence stratum, still ~20% |
| By evidence: Table (`33`) | 0.0833 | 0.0833 | Severe bottleneck |
| By evidence: Layout (`14`) | 0.0652 | **0.0000** | Severe bottleneck |
| By evidence: Chart (`6`) / Figure (`13`) | 0.0000 / 0.0702 | 0.0000 / 0.0769 | Vision-gated |
| By format: Str (`16`) / Int (`20`) / Float (`29`) / List (`17`) | 0.2914 / 0.1000 / 0.0690 / 0.1127 | 0.2344 / 0.1500 / 0.0690 / 0.0588 | Float = 6.9% on both |
| Cited file == gold `doc_id` | 33 / 33 (100%) | 36 / 41 (87.8%) | Gemini accepted wrong-doc answers (e.g., `q0140`) |
| Outcomes (`qwen` / `none` / `check_failed`) | 20 / 76 / 13 | 21 / 66 / 22 | Gemini triggered +9 more `check_failed` |

### 1.2 Where the 62 No-Answer Dev Rows Fail (`diagnose-dev.md`, `POTENTIAL-IMPROVEMENTS-PARKED.md`)
1. **Shared-corpus non-self-contained ceiling (`R0`):** Qwen judged 60/82 in-scope dev Qs not self-contained (43/62 no-answer + 17/20 accepted). Hand audit (`docs/POTENTIAL-IMPROVEMENTS-PARKED.md:136-141`) confirmed **27/43 R0 rows (33% of 82)** are genuinely unanswerable without knowing which PDF ("this report", "page 3"), while **16/43** actually name the entity/topic (Nike, Netflix, NTU, Godfrey Phillips, ProgramFC, HOVER) and are true retrieval/computation misses.
2. **Table retrieval & derived numeric computation (`R1=13`, `R2=3`):** **11/13 R1** and **3/3 R2** are Table/Float questions (`diagnose-dev.md:17-34`). Examples: Costco FY2021 debt/EBITDA (`q1018`), Nike FY2021 inventory/receivables/payables turnover & cash conversion cycle (`q1041-1045`), Netflix 2015 personnel-cost delta (`q1032`). Gold numbers do not appear verbatim in the PDF—they require retrieving **all** relevant statement pages (often 2–4 pages) and computing ratios/differences. `per_big=4` + 20 `grep_lines` misses one of the input pages; `ask_documents.py:938-960` already has `_eval_calc`, yet rejects if any operand is missing from passages or expression fails strict AST/form checks (`2/13 check_failed`).
3. **Spelling/unit false rejections (`_key_absent` = 8/13 `check_failed`, P6):** `_key_tokens` (`ask_documents.py:691-712`) extracts capitalized mid-question words (`FY2021`, `FY2023`, `Bestbuy`, `Rupee`, `Hotels`) and rejects verbatim-correct answers when the PDF writes `fiscal 2021`, `Best Buy`, or currency symbols (`q1025`, `q1083`, `q0939`). Also `names_numbers_ok` caused `3/13 check_failed`.
4. **Two zero-chunk image-only slide decks:** `digitalmeasurementframework...pdf` (196 pp) and `indonesiamobilemarketresearch...pdf` (38 pp) yielded **0 chunks** (`diagnose-dev.md:162-165`, `diagnose-scanned-pages.json:93-352`), making 5 dev rows impossible without OCR (and `--ocr` on `.zip` is currently broken at `ingest.py:660,781`, `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md:13,47`). **Note:** `0/13` R1 rows were caused by scanned pages in the remaining 23 PDFs (`diagnose-dev.md:45-47`).

**Crucial takeaway:** Because Gemini Flash-Lite scored *worse* than Qwen3.8-27B (11.89% vs 12.90%), upgrading the LLM will NOT fix ME4. The bottlenecks are shared-corpus routing, `per_big=4`/grep passage recall on multi-page tables, missing/fragile computation + formatting, and overly brittle deterministic post-checks (`_key_absent`, `names_numbers`).

---

## 2. Top Published Papers, Technical Reports & Leaderboard Entries (14 Entries)

*Note: Standard MMLongBench-Doc (V1) uses 1 PDF per question (`1,082` Qs in NeurIPS 2024 paper; `1,091` Qs in HF/GitHub update; `1,071` Qs in V2). Unless marked otherwise, results below are single-PDF V1.*

### 2.1 Summary Comparison Table

| # | Entry / Paper | Date | Headline Acc / F1 (Scope) | Model(s) | Input Mode | Status |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Nemotron 3.5 Super VL** (BenchLM) | 2026 | **62.8%** Acc (lab-reported) | Nemotron 3.5 Super VL (120B A12B) | Vision (Images) | `UNVERIFIED` |
| 2 | **Qwen3.5 / Gemini Pro** (BenchLM) | 2026 | **~61.5%** / **~60.5%** Acc | Qwen3.5 / Gemini Pro | Vision (Images) | `UNVERIFIED` |
| 3 | **SimpleDoc** (EMNLP 2025) | 2025 | **60.58%** Acc; **62.22%** pg-F1 (Full) | Qwen3-30B + Qwen2.5-VL-32B | Vision + Text summaries | `UNVERIFIED` |
| 4 | **VistaTree + Claude** | 2025–26 | **56.8%** F1 (Full); **43.5%** F1 local | Local VLM tree + Claude reader | Vision-derived tree + pages | `UNVERIFIED` |
| 5 | **MDocAgent** (`arXiv:2503.13964`) | Mar 2025 | **55.3%** Acc (top-6 Qwen); **~42.0%** (GPT-4o) | Qwen3-30B + Qwen2.5-VL-32B / GPT-4o | Dual: Text (ColBERTv2) + Image (ColPali) | `UNVERIFIED` |
| 6 | **MAGE-RAG** (`arXiv:2606.15906`) | Jun 2026 | **53.26%** Acc / **51.19%** F1 (Full) | Agentic LVLM + graph | Vision + Graph elements | `UNVERIFIED` |
| 7 | **MM-Doc-R1** (ACL Findings 2026) | Apr–Jul 2026 | **~49.7 / 46.1**; +10.4% vs baselines | Qwen3-8B + SPO RL | Vision + Text/TOC/Table tools | `UNVERIFIED` |
| 8 | **GPT-4o** (Ma et al., NeurIPS 2024) | Jul 2024 | **44.9%** F1 (NeurIPS) / **42.7%** (v1) | GPT-4o | Vision (Page screenshots) | `VERIFIED` (repo + prior) / `UNVERIFIED` (v1) |
| 9 | **M3DocRAG** (`arXiv:2411.04952`) | Nov 2024 | **41.8%** Acc (top-2/6); **39.7%** (top-10) | ColPali + Qwen2-VL | Vision (Page images) | `UNVERIFIED` |
| 10 | **GPT-4V** (Ma et al., NeurIPS 2024) | Jul 2024 | **30.5%** F1 (NeurIPS) / **31.4%** (v1) | GPT-4V / GPT-4-turbo | Vision (Page screenshots) | `VERIFIED` (repo + prior) / `UNVERIFIED` (v1) |
| 11 | **Docopilot-8B** (CVPR 2025) | Jul 2025 | **28.8%** Acc / **23.0%** F1 (Full) | Docopilot-8B (native doc MLLM) | Vision (Native pages) | `UNVERIFIED` |
| 12 | **Mixtral-8x22B + OCR** (Ma et al.) | Jul 2024 | **~25.0%** F1 (Full, **best orig. text-only**) | Mixtral-8x22B + Tesseract OCR | **Text-only (OCR)** | `UNVERIFIED` |
| 13 | **TRIDENT** (*What the Reranker Sees*) | 2025–26 | **F1@5: 0.332** (ColPali) / **0.254** (BGE-M3); **+6.6** synth | LLM reranker + VLM (`Trident-S`) | Text semantic records + VLM | `UNVERIFIED` |
| 14 | **PageIndex Flash** (OSS-Bench / V2) | Aug 2026 | **85.5%–100%** Acc (**62 pure-text lookup Qs**, 34 PDFs) | PageIndex Flash + LLM | **Text-only (PDF layout tree)** | `UNVERIFIED` |

*(Also notable: **LAD-RAG**, `arXiv:2510.07233` / ACL 2026, reports >90% perfect retrieval recall and +20% recall over BM25/E5/BGE/ColPali at matched noise without static top-$k$ tuning; `UNVERIFIED (search_web, 2026-11-28)`.)*

### 2.2 Detailed Method Breakdown (Top Entries)

*(All external V1 methods below use the official GPT-4o typed-answer extractor + rule-based scorer unless noted; all `search_web` entries accessed 2026-11-28.)*

1. **GPT-4o & GPT-4V (Ma et al., NeurIPS 2024, `arXiv:2407.01523`, `https://github.com/mayubo2333/MMLongBench-Doc`)** — `VERIFIED (repo + prior primary)` for 44.9%/42.7% & 30.5%/31.4%; `UNVERIFIED (search_web, 2026-11-28)` for details.
   - **Score & Scope:** GPT-4o **44.9% F1** (NeurIPS, 1,082 Qs) / **42.7% F1** (v1); GPT-4V **30.5% F1** (NeurIPS) / **31.4% F1** (v1); Human ~66.0% F1.
   - **Method:** Vision (all PDF page screenshots concatenated; no retrieval unit, no query rewrite, no rerank); in-LLM CoT for numeric computation; prompted abstention ("Not answerable"); GPT-4o typed extractor (`Str/Int/Float/List/Not answerable`).
   - **Cost/Ablation:** Very high token cost (~47.5 pages/Q); aggressive answering boosted recall but increased false positives on the 20.6% unanswerable set.

2. **Mixtral-8x22B + Tesseract OCR (Ma et al., NeurIPS 2024, `arXiv:2407.01523`)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **~25.0% F1** on full set — **best published text-only baseline in the original paper**, beating 12 of 14 tested LVLMs (incl. Gemini-1.5-Pro ~20.9% F1, Claude-3-Opus ~18.6% F1, open-source LVLMs <15% F1).
   - **Method:** Text-only via Tesseract OCR over PDF pages; window-bounded text prompt (no visual charts/layout); in-LLM arithmetic/abstention; official GPT-4o extractor.
   - **Takeaway:** Text-only + lossy OCR beat almost all early native LVLMs because multi-image context dilution hurt LVLMs more than OCR loss hurt text LLMs.

3. **SimpleDoc (EMNLP 2025, `aclanthology.org` / `arxiv.org`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **60.58% Acc** (top-30 pool, avg 3.46 pages kept) / **59.55% Acc** (top-10 pool, avg 3.19 pages) on full MMLongBench-Doc; **62.22% page-level F1** (67.37% all-hit rate).
   - **Model & Input:** Vision + text summaries; ColQwen-2.5 retriever + Qwen3-30B + Qwen2.5-VL-32B.
   - **Method:** Page-level retrieval (`top-10`/`top-30` pool) -> **dual-cue summary-based rerank** -> single-VLM **iterative refinement** (up to 3 rounds) with dynamic working memory generating follow-up search queries when evidence is incomplete; numeric/abstention handled inside VLM working memory; official formatting/scorer.
   - **Cost/Ablation:** Keeps only ~3.2–3.5 pages vs 12–20 in MDocAgent. Summary rerank lifted Page F1 **38.75% -> 61.42% (+22.67 pts)**; pool top-2 -> top-30 lifted Acc **56.66% -> 60.58% (+3.92 pts)**.

4. **MDocAgent (Han et al., Mar 2025, `arXiv:2503.13964`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **55.3% Acc** (top-6, 12 pages retrieved, Qwen3-30B + Qwen2.5-VL-32B in SimpleDoc Table 1; 50.6% top-4; 54.8% top-10); ~42.0% with GPT-4o (GitHub README). Reported +12.1% average gain across benchmarks.
   - **Model & Input:** Dual-track: Text (**ColBERTv2**) + Image (**ColPali**); Qwen or GPT-4o agents.
   - **Method:** Page/chunk dual retrieval -> 5 specialized agents (General, Critical noise-filter, Text Agent, Image Agent, Summarizing Agent reconciling text+image); no external calculator reported; Summarizing Agent decides abstention & formats answer.
   - **Cost/Ablation:** High multi-agent call count (5 agents/Q). Removing Text/Image or General/Critical agents drops accuracy; scaling $k$ from 6 to 10 (12 -> 20 pages) *drops* Acc 55.3% -> 54.8% (context dilution).

5. **MAGE-RAG (Jun 2026, `arXiv:2606.15906v1`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **53.26% Acc / 51.19% F1** on full MMLongBench-Doc (52.75% on LongDocURL).
   - **Method:** Offline multigranular evidence graph (page + element nodes: text, tables, charts, images; containment, reading order, layout adjacency, section hierarchy, semantic-neighbor edges) -> online page-entry + budgeted controller (`activate/open/search/prune` under $(B_p, B_a, B_o, B_T, B_c)$) -> page-organized LVLM reader.
   - **Cost/Ablation:** Explicit budget vector bounds token/latency cost; ablating graph expansion or dynamic search hurts cross-page queries, while unbounded budgets inject noise.

6. **MM-Doc-R1 (Apr 2026 / Findings of ACL Jul 2026, `arxiv.org` / `aclanthology.org`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** +10.4% average gain over prior baselines (~49.7 / 46.1 with Qwen3-8B+SPO).
   - **Method:** Multi-turn vision-aware planner uses TOC/structure to invoke table parsers, layout readers, and page retrieval iteratively; trained with Similarity-based Policy Optimization (SPO) to fix GRPO multi-turn baseline bias; in-trajectory tool/model reasoning.
   - **Ablation:** SPO over GRPO adds **+5.0%** (Qwen3-8B) and **+6.1%** (Qwen3-4B); removing TOC-guided planning drops score (~44.3).

7. **VistaTree / VistaText (2025–2026, `arxiv.org`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **43.5% F1** (flat selection, local VLM) / **42.1% F1** (hierarchical); **56.8% F1** with Claude reading local tree; gold-page oracle = **61.0% F1** on answerable.
   - **Method:** Segments PDF into page spans, builds image-derived topic hierarchy offline with sibling summary refinement; flat leaf selection or hierarchical navigation at query time; Claude/VLM reads selected pages.
   - **Cost/Ablation:** Cuts input tokens **73.8%** vs full-PDF feeding; flat global visibility beats hierarchical traversal (+1.4 F1).

8. **M3DocRAG (Cho et al., Nov 2024, `arXiv:2411.04952`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **41.8% Acc** (top-2/6), **39.7% Acc** (top-10) on MMLongBench-Doc (SimpleDoc Table 1).
   - **Method:** Single-pass ColPali page-image retrieval (top-$k$) -> Qwen2-VL VLM (no query rewrite, no rerank, no external calc); official extractor/scorer.
   - **Ablation:** Exceeding top-6 pages drops accuracy (41.8% -> 39.7%).

9. **Docopilot-8B (CVPR 2025, `arXiv:2507.14675`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
   - **Score & Scope:** **28.8% Acc / 23.0% F1** (8B); **21.8% Acc / 16.0% F1** (2B) on full set (vs InternVL2-8B 17.4 Acc / 16.5 F1; InternVL2-8B+RAG 24.2 Acc / 24.5 F1).
   - **Method:** Native document-level MLLM (no external retrieval); single pass over pages -> official extractor.
   - **Latency/Ablation:** Lower inference latency than multi-stage RAG, but lags reranked/agentic RAG by >25–30 pts.

10. **TRIDENT (*What the Reranker Sees*, 2025–2026, `arxiv.org` / `openreview.net`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
    - **Score & Scope:** BGE-M3 text retrieval gets **Recall@20 = 0.86** but **F1@5 = 0.254** (ColPali **F1@5 = 0.332**); `Trident-S` adds **+6.6 pts** on open-ended synthesis.
    - **Method:** `Trident-R` enriches candidate pages into structured **semantic records** (section path, entity tags, visual captions, concept hits, snippets) for 1 adaptive-$K$ LLM rerank; `Trident-S` prompts synthesis via topical/entity/structural lenses.
    - **Ablation:** LLM reranking on raw text snippets fails to improve MRR/F1; structured page annotations drive the selection gain.

11. **LAD-RAG (Oct 2025 / ACL 2026, [`https://arxiv.org/abs/2510.07233`](https://arxiv.org/abs/2510.07233), [`https://aclanthology.org/2026.acl-long.724/`](https://aclanthology.org/2026.acl-long.724/), accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
    - **Score & Scope:** **>90% perfect retrieval recall** across MMLongBench-Doc/LongDocURL/DUDE/MP-DocVQA without static top-$k$ tuning (**+20% recall** over BM25/E5/BGE/ColPali at matched noise).
    - **Method:** Builds layout-aware symbolic document graph + neural embeddings; dynamic LLM agent navigates graph rather than fixed top-$k$.

12. **PageIndex Flash & MMLongBench-Doc-V2 (Zhang / VectifyAI, Aug 2026, `arXiv:2608.03397`, `pageindex.ai`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
    - **Score & Scope:** V2 fixes 106 annotations -> 1,071 Qs / 134 PDFs and replaces string match with a pinned LLM judge (explicitly non-comparable to V1). On `PageIndex-OSS-Benchmark` (**62 pure-text lookup Qs, 34 PDFs, 1,945 pp** from V2), **PageIndex Flash** gets **85.5%–100% Acc**.
    - **Method:** **Text-only** PDF text-geometry -> hierarchical TOC/tree index (no OCR/vision at index time) -> LLM tree navigation -> targeted page reading; no external calculator.
    - **Cost/Latency:** Indexing takes **13 s – 4.5 min** (9 to ~1,100 pp) at **~$0.001/page**; cuts query tokens **38%–91%** vs flat context stuffing.

13. **Frontier 2026 Native VLMs (`benchlm.ai`, accessed 2026-11-28)** — `UNVERIFIED (search_web, 2026-11-28)`.
    - **Scores:** **Nemotron 3.5 Super VL (120B)** **62.8%** (lab-reported), **Qwen3.5** ~**61.5%**, **Gemini Pro** ~**60.5%**, **Kimi K2.5** **58.5%**, **Nemotron 3 Nano Omni (30B)** **57.5%**. High accuracy via massive long-context vision backbones; infeasible on Pi 4.

---

## 3. Ablation Evidence: What Moved the Score and by How Much

| Technique | Paper / Source | Baseline -> With Technique | Delta on MMLongBench-Doc | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Summary-based dual-cue page reranking** | SimpleDoc (EMNLP 2025) | ColQwen-2.5 top-2 (38.75% pg-F1) -> dual-cue rerank (61.42% pg-F1) | **+22.67 pts Page F1** | `UNVERIFIED (search_web, 2026-11-28)` |
| **Layout-aware symbolic graph + dynamic retrieval** | LAD-RAG (`arXiv:2510.07233`) | Static BM25/E5/BGE/ColPali -> dynamic graph | **+20 pts recall** (>90% perfect recall) | `UNVERIFIED (search_web, 2026-11-28)` |
| **Multi-turn RL planner + TOC/table tools (SPO vs prior)** | MM-Doc-R1 (ACL Findings 2026) | Prior baselines -> MM-Doc-R1; GRPO -> SPO | **+10.4% avg** (+5.0% Qwen3-8B, +6.1% Qwen3-4B) | `UNVERIFIED (search_web, 2026-11-28)` |
| **Multi-lens synthesis (`Trident-S`)** | TRIDENT (*What the Reranker Sees*) | Single-lens VLM -> topical/entity/structural lenses | **+6.6 pts** (open-ended synthesis) | `UNVERIFIED (search_web, 2026-11-28)` |
| **Expanding retrieval candidate pool (before rerank)** | SimpleDoc (EMNLP 2025) | Top-2 pool (56.66% Acc) -> Top-30 pool (60.58% Acc) | **+3.92 pts Acc** | `UNVERIFIED (search_web, 2026-11-28)` |
| **Flat summary selection vs hierarchical tree walk** | VistaTree (`arXiv`) | Hierarchical (42.1% F1) -> Flat leaf selection (43.5% F1) | **+1.4 F1** (global map beats narrow tree walk) | `UNVERIFIED (search_web, 2026-11-28)` |
| **Iterative follow-up query refinement (1 -> 3 rounds)** | SimpleDoc (EMNLP 2025) | 1 iter (58.62% Acc) -> 3 iters (59.55% Acc) | **+0.93 pts Acc** | `UNVERIFIED (search_web, 2026-11-28)` |
| **Blindly increasing static `top-k` pages** | MDocAgent / M3DocRAG | MDocAgent k=6 (55.3%) -> k=10 (54.8%); M3DocRAG k=6 (41.8%) -> k=10 (39.7%) | **-0.5 to -2.1 pts Acc** (context dilution!) | `UNVERIFIED (search_web, 2026-11-28)` |
| **Swapping Qwen3.8-27B -> Gemini Flash-Lite in ME4** | Repo (`score-dev*.md`) | Qwen (12.90% headline / 0.2095 F1) -> Gemini (11.89% / 0.1912 F1) | **-1.01 pts Acc / -0.018 F1** | `VERIFIED (repo)` |

---

## 4. What a Text-Only Lane Can Get

1. **Full-benchmark text-only ceiling:** On all 1,082–1,091 MMLongBench-Doc questions, **~16–20% require pure visual Chart/Figure reading** (plus visually-encoded layout cues) and are unreachable without vision (`docs/MMLONGDOC-BENCH.md:128,147-148`; in ME4's 25-doc sample, 19/109 dev rows have `Chart` or `Figure` evidence, `score-dev.md:15-16`). Ma et al.'s original **Tesseract OCR + Mixtral-8x22B** baseline scored **~25.0% F1** overall (`UNVERIFIED (search_web, 2026-11-28)`). Even with ideal page selection (gold-page oracle, `VistaTree`), answerable-question F1 tops out around **61.0% F1** *with* a strong VLM (`UNVERIFIED`); for a **text-only** lane on the full single-PDF task, realistic overall F1 is **~30%–40%**, bounded by ~20% vision-only questions and table/layout parsing loss.
2. **Pure-text lookup ceiling (when text is preserved):** **PageIndex Flash** (`PageIndex-OSS-Benchmark`, 62 pure-text lookup Qs across 34 PDFs / 1,945 pages) shows that **85.5%–100%** accuracy is achievable *without* vision or embeddings if PDF layout/TOC structure is preserved (`UNVERIFIED (search_web, 2026-11-28)`). Similarly, **TRIDENT** shows text retrieval (**BGE-M3**) already achieves **0.86 Recall@20** on MMLongBench-Doc—text retrieval finds the right neighborhood; the failure is **F1@5 (0.254)** when selecting/reranking pages from raw text snippets (`UNVERIFIED`).
3. **ME4's specific dev ceiling (`25`-PDF shared corpus, `n=82` in-scope answerable):**
   - **Hard upper bound today:** `27/82` (33%) are hand-verified non-self-contained across 25 PDFs (`POTENTIAL-IMPROVEMENTS-PARKED.md:137-141`), and `5` rows belong to 2 zero-chunk image slide decks (`diagnose-dev.md:164`) -> **max ~62%–65%** even with an oracle text reader + perfect retrieval (without OCR).
   - **Realistic target for ME4's text-only lane (post-deploy):** Fixing `_key_absent` / `names_numbers` false rejections, switching PDF table extraction to layout-preserving Markdown/text, expanding `per_big` + page-summary/TOC routing, enabling 2nd-hop fallback, and tightening numeric/calc handling should move ME4 from **12.9% -> ~30%–42%** on the shared-corpus in-scope answerable set (and **~50%–65%** if evaluated in 1-PDF-per-question mode matching the literature).

---

## 5. Techniques NOT Applicable to ME4 (and Why)

| Published Technique | Why NOT Applicable to ME4 | Exception / Ingest-Only Caveat |
| :--- | :--- | :--- |
| **ColPali / ColQwen-2.5 page-image embeddings** (M3DocRAG, MDocAgent, SimpleDoc) | Requires GPU + GBs of multi-vector page patch embeddings; Pi 4 (4 GB RAM, CPU-only) cannot encode or score late-interaction vision patches within ~1–3 s latency. | None at query time. |
| **End-to-end / Page-image VLM readers** (GPT-4o, Qwen2.5-VL-32B, Nemotron 3.5, Docopilot) | Violates ME4's text-only query constraint (`mmld-context.txt:18-19`: no vision model on Pi, spoken answers, Flash-Lite text/JSON lane). | Offline laptop-ingest *could* caption charts/slide decks into text at ingest time, though out of scope for pure-text PDF parsing. |
| **VistaTree image-derived summary trees** | Uses VLM calls on rendered page images to build summaries. | **Text-only analog IS applicable:** build TOC/section-summary trees from extracted PDF text/headings at ingest time (`PageIndex Flash` style). |
| **MDocAgent 5-agent + 12–20 page orchestration** | Multiple sequential LLM + VLM calls per query blows Pi latency budget (few seconds to first real audio) and paid API budget. | A single lightweight 2nd-hop fallback (`_maybe_fallback`, already in `ask_documents.py`) is the bounded ME4 equivalent. |
| **MM-Doc-R1 SPO RL fine-tuning** | Requires training open-weight 4B–8B reasoning models; ME4 uses frozen Gemini Flash-Lite API in production. | Not applicable. |

---

## 6. Shared Corpus vs. One PDF Per Question

* **Official MMLongBench-Doc is strictly 1 PDF per question** (`docs/MMLONGDOC-BENCH.md:10,161`).
* **What the literature does:** Follow-ups almost universally preserve 1 PDF per question on MMLongBench-Doc. When testing multi-document retrieval, **M3DocRAG** (`arXiv:2411.04952`) created a *separate* open-domain benchmark (**M3DocVQA**, 3,368 PDFs) with self-contained questions rather than pooling MMLongBench-Doc's 135 PDFs (`UNVERIFIED (search_web, 2026-11-28)`). Likewise, **PageIndex-OSS-Benchmark** pooled 34 PDFs (1,945 pages) from MMLongBench-Doc-V2 *only* for 62 self-contained lookup questions (`UNVERIFIED`).
* **What pooling 25 PDFs did to ME4 (`VERIFIED (repo)`):**
  1. **Unfixable R0 ceiling (~33%+):** At least **27/82 (33%)** of dev in-scope questions contain anaphoric references (*"this report"*, *"the second page"*, *"table of contents"*) with zero document identifier (`POTENTIAL-IMPROVEMENTS-PARKED.md:136-141`). No retriever in the world can disambiguate *"What date is mentioned on the second page?"* (`q0237`) across 25 PDFs.
  2. **Cross-document distractor pollution:** Five 10-Ks (Amazon, BestBuy, Costco, Netflix, Nike) share identical financial terminology (*"long-term debt"*, *"EBITDA"*, *"repurchase"*). Qwen maintained **33/33** cited-file match by refusing uncertain hits (`76 none`), whereas Gemini answered **36/41** (`5 wrong-doc accepts`, e.g., `q0140` answered Nebraska Historic Buildings Survey for `efd88e41...`, `run-dev-gemini.jsonl:2`).
  3. **Benchmark implication:** Keep the 25-doc shared corpus to measure **real ME4 cross-file routing** (on the self-contained subset), **AND** add a `--doc-scoped` mode in `scripts/bench_mmlongdoc.py` (filtering `file_context` to `row["doc_id"]`) so ME4's reading/table/calc improvements can be compared cleanly against single-PDF literature without the 33% R0 floor masking real gains.

---

## 7. Ranked Techniques to Apply to ME4 (Expected Value / Effort × Risk)

*Ranked for ME4's post-deploy branch under Pi 4 (4 GB RAM, CPU-only, <3 s query latency) + Gemini Flash-Lite constraints.*

### Summary Ranking Table

| Rank | Technique | Literature Backing | Expected Value | Effort | Pi Cost (Query / Ingest) | Risk | Primary Bottleneck Hit |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **1** | **Fix `_key_absent` normalization + `file_has` wiring** | Repo P6 (`diagnose-dev.md`) | **Very High** | **XS** | ~0 ms / 0 RAM | Low | **8/13 `check_failed`** (`FY2021` vs `fiscal 2021`, `Bestbuy` vs `Best Buy`, units) |
| **2** | **Enable 2-step catalog/query-rewrite fallback (`allow_fallback=True`)** | SimpleDoc (+0.93 to +3.92), MM-Doc-R1 | **High** | **XS–S** | +1 LLM call *only* on `none` | Low–Med | Recovers missed docs/pages on self-contained R1/R0-miss rows |
| **3** | **Structured page-index / TOC-card + multi-aspect summary rerank** | SimpleDoc (**+22.7 pg-F1**), VistaTree (**-73.8% tok**), TRIDENT, PageIndex | **Very High** | **M** | +1 tiny prompt or 0 extra calls; small JSON tree in SQLite | Low | Fixes `per_big=4` + `grep_lines` blindness on 47–109p PDFs; locates exact statement pages |
| **4** | **Layout-preserving PDF table extraction (`pdfium` + column/row structure -> Markdown/TSV)** | LAD-RAG (+20% recall), MAGE-RAG | **High** | **M** | 0 ms query / slight ingest time (offline or Pi) | Low–Med | Fixes Table/Float failures (**11/13 R1, 3/3 R2**) where column misalignment breaks values |
| **5** | **Multi-page table bundle + resilient `calc` & `names_numbers` relaxation** | Repo `diagnose-dev.md` (14/16 Table/Float) | **High** | **S–M** | +1–2 chunks context (~500 tok) | Med | Unlocks derived ratios/deltas (`calc` 2/13, `names_numbers` 3/13) |
| **6** | **Fix `.zip --ocr` + ingest-time OCR on zero-chunk PDFs** | Repo T5 (`BENCHMARK-PROTOCOL-IDEATION`) + Ma et al. OCR | **Medium** | **S–M** | 0 ms query / +ingest CPU (bounded) | Low | Unlocks 2 zero-chunk slide decks (**5 dev rows**) + 242 scanned pages |
| **7** | **Optional `--doc-scoped` eval arm + self-contained dev slice in `bench_mmlongdoc.py`** | Repo `diagnose-dev.md` R0 audit | **High (diagnostic)** | **XS** | Harness only (0 Pi impact) | None | Separates retrieval routing from reading/calc ceiling |

### Detailed Action Specs & Exact Harness Measurement

1. **Rank 1 — Fix `_key_absent` spelling/unit rejections & pass `file_has` in harness**
   - *Change:* In `ask_documents.py:691-745`, fold `fy(\d{4})` <-> `fiscal \1`, allow space-collapsed matching (`bestbuy` <-> `best buy`), and skip standalone currency/unit words (`rupee`, `dollar`, `million`) in `_key_tokens`. Also verify `file_has` is passed consistently in `bench_mmlongdoc.py:1184`.
   - *Cost/Risk:* Zero Pi RAM/CPU; zero extra LLM calls. Slight risk of admitting a wrong-doc match if a key token is dropped—guard with existing `quote_spans` + `names_numbers_ok`.
   - *Measure with existing harness:* Re-run `scripts/bench_mmlongdoc.py run --split dev --tag dev-r1` and `score --tag dev-r1`. Verify `lane:check_failed` drops from `13` (Qwen) / `22` (Gemini) by >=6–8 rows (`q0939`, `q1025`, `q1083` flip to `accepted`), headline in-scope Acc rises, AND run `scripts/sufficiency_eval.py` (100-row set) to confirm zero false-positive regressions on document-only controls.

2. **Rank 2 — Activate bounded 2nd-hop fallback & query rewriting (`_maybe_fallback`)**
   - *Change:* Set `allow_fallback=True` and pass `catalog` in `bench_mmlongdoc.py:1181-1187` (matching production), and upgrade `FALLBACK_SYSTEM` (`ask_documents.py:169-174`) so on `status=="none"` it can emit **both** a target file AND refined search keywords (e.g., expanding *"inventory turnover"* -> *"cost of sales inventories balance sheet"* for Nike 10-K).
   - *Cost/Risk:* +0 calls when 1st pass succeeds; +1 Flash-Lite call (~0.5–1.0 s) only on `none` misses. Medium risk in shared corpus of grabbing a wrong distractor file unless `cited_file == doc_id` is monitored.
   - *Measure with existing harness:* Compare `run --split dev --tag dev-r2` vs baseline: track `calls=2` recovery count, `cited file == doc_id` rate (must stay >90%), and `Table` + `Pure-text` accuracy deltas.

3. **Rank 3 — Page-Index / Section-TOC tree & Page-Summary Reranking (`SimpleDoc` + `PageIndex` text-only adaptation)**
   - *Change:* At ingest time (`src/voice_agent/rag/ingest.py`), extract a compact **per-PDF page/section map** (page number -> heading/first lines + 1-line keyword/entity summary, stored in SQLite `file_cards` / `page_index` table—no embeddings, no vision). At query time (`dbindex.py:file_context`), for big PDFs (>15 KB), instead of blind `per_big=4` FTS chunks + 20 `grep_lines`, pass the target file's compact page-TOC index (or use FTS top-30 chunks -> page-summary ranker) to pick the best **4–6 pages** (including adjacent page pairs like p.59+p.61 for financial statements!).
   - *Cost/Risk:* Small SQLite metadata footprint (<50 KB/PDF); 0 GPU; query time adds <50 ms SQLite lookup (or 1 tiny Flash-Lite page-picker call on >40-page PDFs). Replaces blind chunk fragmentation without violating Pi 4 limits.
   - *Measure with existing harness:* `bench_mmlongdoc.py run --split dev --tag dev-r3` + `diagnose` script: measure `R1` drop (currently `13/62`, **11** in `Table`!), `Table` Acc (currently `0.0833`), and `Layout` Acc (`0.0652`).

4. **Rank 4 — Layout-preserving PDF table extraction at ingest (`pdfium` geometry -> Markdown/aligned rows)**
   - *Change:* In `src/voice_agent/rag/readers.py`, reconstruct table rows/columns using `pypdfium2` text-box `(x, y)` coordinates (group by `y` tolerance, separate columns by `x` gap with `|` or `\t`, repeat multi-line headers) so row labels stay glued to column numbers within the same chunk (solving Ticket 275's split-chunk table loss).
   - *Cost/Risk:* 0 ms query latency; modest CPU increase at ingest (still pure `pypdfium2`, zero ML weights). Risk of mis-grouping multi-column prose—apply table-grid formatting only when `x`-aligned column gaps repeat across >=3 consecutive lines.
   - *Measure with existing harness:* Rebuild `out/mmlongdoc/corpus25/corpus.db` via `bench_mmlongdoc.py ingest --force`, run `scripts/bench_mmlongdoc_extraction.py` (Ticket 275) to verify Table integrity pass rate increases, then run `bench_mmlongdoc.py run/score --split dev --tag dev-r4` and check `Table` / `Float` accuracy.

5. **Rank 5 — Multi-page table context + resilient `calc` & `names_numbers` verification**
   - *Change:* Allow `file_context` (`dbindex.py:391`) to pull `per_big=6` chunks (or full page text for top-2 pages of a big PDF) so multi-statement ratios (e.g., COGS on Income Statement + Inventory on Balance Sheet) co-occur in context. In `ask_documents.py:938-965`, normalize thousands/millions scaling and rounding tolerance (`±1` in last decimal place) between `calc_val` and `answer` (e.g., `12.35` vs `12.346`), and allow intermediate ratios when operands are verified in quotes.
   - *Cost/Risk:* +1–2 KB prompt tokens per big-PDF turn (~100–200 ms Flash-Lite latency). Guard against arithmetic hallucination by keeping `calc_nums in passage_nums` strict.
   - *Measure with existing harness:* `bench_mmlongdoc.py run/score --split dev --tag dev-r5`: track `by answer_format: Float` (currently **0.0690**) and `check_failed: calc` (`2/13`) + `names_numbers` (`3/13`).

6. **Rank 6 — Fix `.zip --ocr` & run bounded RapidOCR on zero-chunk / scanned pages (T5)**
   - *Change:* Fix `ingest.py:660,781` `work_dir` cleanup bug + `images.py:147-149` `px^2` vs `pt^2` threshold + `ocr.py:360` `filter_lines` (already scoped as T5 in `docs/BENCHMARK-PROTOCOL-IDEATION-2026-11-27.md:112`). Run `--ocr` on the 25-PDF corpus (unlocking the 2 image-only slide decks = 5 dev rows).
   - *Cost/Risk:* 0 ms query cost; high *ingest* CPU on Pi if run on full 196-page slide decks (keep `ocr_deadline_min` cap or run OCR on laptop before shipping DB).
   - *Measure with existing harness:* Re-ingest with `--ocr`, inspect `diagnose-scanned-pages.json` / DB chunk counts on the 2 empty PDFs, then `run/score --split dev --tag dev-r6` specifically checking those 5 dev rows (`q0251, q0264, q0275, q0277, q0280`).

7. **Rank 7 — Diagnostic harness split: `--doc-scoped` + self-contained slice (no product change)**
   - *Change:* Add an optional `--doc-scoped` flag in `bench_mmlongdoc.py` (restricting `file_context` to `row["doc_id"]`) and report both **shared-corpus (self-contained)** and **single-PDF** scores side by side.
   - *Cost/Risk:* Harness-only, zero production or Pi impact. Tune exclusively on `dev` (`n=109`), keep `test` (`n=91`) untouched until final post-deploy A/B.
   - *Measure with existing harness:* Run baseline `dev` under `--doc-scoped` to quantify the exact R0 shared-corpus penalty vs ME4's true reading/table/calc accuracy.
