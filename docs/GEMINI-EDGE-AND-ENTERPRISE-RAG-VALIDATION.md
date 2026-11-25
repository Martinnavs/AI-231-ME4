# Empirical Validation and Hardening of Edge-to-Cloud Voice, RAG, and Agentic Architectures

The core architectural thesis in your MLOps Projects stack—decoupling ultra-lightweight edge execution from selectively escalated cloud reasoning, backed by deterministic programmatic gates rather than unconstrained LLM loops—is **well supported** by 2024–2026 peer-reviewed studies and hardware benchmarks across both edge devices and large-scale enterprise deployments. External evidence corroborates your rejection of vanilla Whisper on Raspberry Pi 4, your use of Sherpa-ONNX / Parakeet-TDT and Piper-class neural TTS for CPU cascades, your avoidance of intrinsic LLM self-correction, your tail-placement of dynamic chain state to preserve prefix caches, and your decision to keep credentials outside the network-enabled sandbox.

At the same time, external research highlights **five material gaps** in the current specification that can undermine production reliability unless addressed:

1. **XML delimiting is insufficient for indirect prompt injection:** Wrapping untrusted web or RAG outputs in XML delimiters (`<untrusted_search_data>`) leaves indirect prompt injection attack success rates near 30% because fixed delimiters are easily escaped; upgrading to **datamarking** and structural isolation lowers residual attack success to 1–5% (with an associated utility trade-off).
2. **Naive RAG and raw BM25 thresholds hurt truthfulness:** Uncalibrated retrieval and unnormalized BM25 score cutoffs frequently cut net truthfulness by inflating hallucinations even when raw recall increases.
3. **NLI/span hallucination classifiers fail as hard automated blockers:** Standalone grounding detectors—including MiniCheck and LettuceDetect—exhibit severe operating-point false-positive rates (flagging 65–99% of valid outputs when tuned for 95% hallucination recall in published audits).
4. **Raspberry Pi 4 systems bottlenecks erase model-level gains:** Thermal throttling (2–3x slowdowns without active cooling), A2 microSD card incompatibility, and default SQLite write-serialization settings can negate neural latency optimizations.
5. **Text retrieval degrades numerical table computation:** Retrieval helps prose-heavy extraction but systematically degrades numerical table computation on lightweight models—including a -19.7% relative drop on Gemini 2.5 Flash-Lite—unless paired with code execution.

## Executive Summary Matrix

| Notebook Architecture Bet | Validation Verdict | Key Verified Research & Production Evidence (Edge & Large) | Required Hardening Action |
| :--- | :--- | :--- | :--- |
| **Reject pruned/vanilla Whisper on RPi 4; use CTC/TDT & compact edge ASR** | **Validated** | Home Assistant guidance estimates vanilla Whisper at ~8 s on RPi 4, and the Vox changelog reports 3.3x RTF for `base.en` due to 30 s zero-padding. Moonshine (RoPE) cuts sub-5 s latency by 5–35x; a community RPi 5 benchmark puts Sherpa-ONNX Parakeet-TDT 0.11B/0.6B at 0.12–0.21 RTF (3.51–4.19% WER). | Keep Parakeet-TDT on x86/Pi 5; evaluate Moonshine Tiny (27M) or Vosk + ONNX tiered intent for standalone RPi 4; enforce active cooling. |
| **Programmatic completion gates (`SOURCES.json`) over LLM self-checking** | **Validated** | Huang et al. (ICLR 2024) demonstrates that intrinsic self-correction drops accuracy across GPT-3.5, GPT-4, and Llama-2 (up to -37.7 pp on CommonSenseQA). | Retain deterministic file/schema checks; evaluate RAG outputs with Meta CRAG's 1/0/-1 truthfulness penalty to expose hallucination surges. |
| **Preserve static 3.4k prompt prefix; place `[ACTIVE TASK]` at user-turn tail** | **Validated** | Liu et al. ("Lost in the Middle") shows GPT-3.5-Turbo drops 22.0 points at middle positions (53.8% vs. 56.1% closed-book). Gemini Flash caches immutable prefixes >=1,024 tokens at up to 90% discount. | Keep system prompt immutable; prune distractors aggressively (Chroma Context Rot reports even one near-miss distractor hurts accuracy). |
| **Focus tracking + selective escalation for multi-turn pronouns (RQ6)** | **Validated with Caveat** | ConvGQR raises QReCC MRR from 10.2 to 42.0 (above human rewrites); a Siri plug-in lifted incomplete follow-ups by +131.57%. TopiOCQA highlights stale "over-carry" errors on topic pivots. | Keep escalation conditional and retain raw-query retrieval branches; track over-carry rate (OC) explicitly on topic switches. |
| **Wrap `web_research` in `<untrusted_search_data>` + host-side `publish_site`** | **Partially Validated (Gap Found)** | Host-side credential isolation aligns with structural defense patterns, but XML delimiting leaves ~30% ASR and is trivially escaped. `--share-net` in `bwrap` exposes loopback services (`127.0.0.1`). | Upgrade delimiting to Spotlighting **datamarking** (3.1% ASR); drop `--share-net` or block `127.0.0.0/8` and `169.254.0.0/16`; add Zip Slip & bomb guards. |
| **Local/Enterprise RAG chunking, BM25 filtering, & graph indexing** | **Needs Strengthening** | GPT-4-Turbo + RAG fails or refuses 81% on FinanceBench; un-reranked hybrid scores below Naive RAG (30.4 vs. 32.9 F1); GraphRAG-Global drops to 10.10% on TimeQA (vs. Vanilla 56.33%). | Mandate per-page OCR triage, header-prepended table chunks, RRF + reranking, interval-aware timestamps, and LiteRAG/LightRAG over GraphRAG. |

---

## Edge Device Validation: Speech Cascades, Compute Ceilings, and Local Storage

### Acoustic Models, STT, and TTS: Confirming the Edge Pareto Frontier

Your MLOps Projects specification rejects transformer-based Whisper models for always-on Raspberry Pi 4/5 listening and standardizes on **Sherpa-ONNX**, **Parakeet-TDT 0.6B**, **MatchboxNet/BC-ResNet CTC**, and **Kitten/Piper-class neural TTS**. External benchmarks and developer telemetry from 2024–2026 largely confirm both the latency penalty of vanilla Whisper and the competitiveness of your chosen CPU runtimes, while identifying newer variable-length and unified-decoder architectures suited for ARM64 boards.

The bottleneck of OpenAI Whisper on short voice commands is architectural rather than strictly a function of parameter count. Vanilla Whisper zero-pads every utterance to a fixed 30-second window (3,000 log-Mel frames downsampled to 1,500 encoder states), forcing quadratic self-attention across empty frames even on a 2-second smart-home command. Approximate local voice guidance from Home Assistant (cited by Turing Pi without full configuration specs) puts Whisper at roughly 8 seconds on a Raspberry Pi 4 versus under 1 second on an Intel NUC. Consistent with that ceiling, the open-source Vox project's Raspberry Pi 4 changelog reports vanilla `whisper.cpp` `base.en` running at 3.3x real-time factor (RTF)—more than three times slower than live speech—requiring Distil-Whisper plus INT8 quantization to reach 0.8x RTF.

#### Benchmark Table 1A: Edge ASR Real-Time Factor (RTF)

*RTF = processing time / audio duration (lower is faster; `< 1.0` = faster than real time).*

| ASR Model / Runtime | RTF | WER | RAM / Footprint | Hardware / Clip Context | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sherpa-ONNX Parakeet-TDT 0.11B** | **0.12** | 4.19% | 1.23 GB RAM | Raspberry Pi 5 | Edge-optimized (`< 1.0`) |
| **Moonshine Tiny (1s audio)** | **0.15** | — | 27M params | 1s utterance (17.7x vs. Whisper Tiny) | Edge-optimized (`< 1.0`) |
| **Sherpa-ONNX Parakeet-TDT 0.6B** | **0.21** | 3.51% | 1.76 GB RAM | Raspberry Pi 5 | Edge-optimized (`< 1.0`) |
| OpenVINO Whisper | 0.29 | 11.31% | 1.39 GB RAM | Raspberry Pi 5 | Intermediate baseline (`< 1.0`) |
| Distil-Whisper + INT8 | 0.80 | — | 4x smaller downloads | Raspberry Pi 4 | Intermediate baseline (`< 1.0`) |
| Whisper Tiny (1s audio) | 2.65 | — | 39M params | 1s utterance (30s padding penalty) | Slower than real time (`> 1.0`) |
| Whisper `base.en` | 3.30 | — | 74M params | Raspberry Pi 4 (baseline prior to Distil+INT8) | Slower than real time (`> 1.0`) |

#### Benchmark Table 1B: CPU-Only Edge TTS Median Single-Inference Latency (P50)

*Lower is faster. Author benchmarks from Piper-Plus (host CPU not specified as RPi 4; verify on target ARM64 silicon).*

| TTS Engine / Variant | Median Latency (P50) | Relative vs. Piper-Plus | Model Notes |
| :--- | :--- | :--- | :--- |
| **Piper-Plus (MB-iSTFT)** | **27 ms** | Baseline | 38 MB (6-lang ONNX), unified MB-iSTFT + PQMF decoder |
| Piper Original (`en_US-lessac-medium`) | 35 ms | Piper-Plus -23% | `rhasspy/piper-voices` v1.0.0 |
| Piper-Plus (HiFi-GAN) | 43.3 ms | Piper-Plus -38% | Previous Piper-Plus decoder baseline |
| Sherpa-ONNX VITS (`vits-piper-en_US-amy-low`) | 53 ms | Piper-Plus -49% | `k2-fsa` release |

Two edge ASR families validate and extend your pipeline depending on target hardware. On **Raspberry Pi 5 and x86 laptops**, a community benchmark on Raspberry Pi 5 supports your Sherpa-ONNX Parakeet-TDT selection: **Parakeet-TDT 0.11B** recorded 4.19% WER at 0.12 RTF (1.23 GB RAM) and **Parakeet-TDT 0.6B** recorded 3.51% WER at 0.21 RTF (1.76 GB RAM), outperforming OpenVINO Whisper (11.31% WER, 0.29 RTF, 1.39 GB RAM) on accuracy, speed, and memory efficiency. On **Raspberry Pi 4 and short sub-5-second utterances**, **Useful Sensors Moonshine** (Jeffries et al., 2024, arXiv:2410.15608) replaces learned positional embeddings with Rotary Position Embeddings (RoPE) so compute scales with actual audio length—processing 200 frames instead of 3,000 for a 2-second command and cutting over 90% of redundant attention math. According to technical summaries of the Moonshine release, Moonshine-Tiny (27M parameters) and Moonshine-Base (61M) reduce latency on sub-5-second audio by 5x to 35x relative to similarly sized Whisper checkpoints—reaching 0.15 RTF versus Whisper Tiny's 2.65 RTF on 1-second audio (a 17.7x speedup), while LiteRT community benchmarks on a Raspberry Pi 5 (`f32`) measure 495.3 ms total latency (0.099 RTF) on a 5-second clip. An independent 2026 study in *Applied Sciences* likewise identified **Moonshine + Qwen 1.5B + Piper** as its optimal constrained-edge voice cascade. Meanwhile, **Sherpa-ONNX v1.13.8** (September 2026) added Intel OpenVINO NPU support alongside RKNN, QNN, and Ascend backends, validating your quick check of local OpenVINO provider availability before ruling it out.

For fully offline smart-home command parsing on a standalone Raspberry Pi 4 without a companion laptop, published maker deployments converge on a three-tier pattern closely resembling your two-stage firewall: a tiny wake-word detector (such as Porcupine or your DS-CNN), **Vosk** (40 MB small or 128 MB `lgraph`, ~200 MB RAM, 300–500 ms response) or **Zipformer-tiny** via Sherpa-ONNX (40–60 MB RAM, 0.8–1.2 RTF, 3–5 W), and a tiered intent classifier (Tier 0 fuzzy phrase cache at ~5 ms; Tier 1 `all-MiniLM-L6-v2` 22 MB ONNX classifier at ~90 MB RAM; Tier 2 LLM fallback at ~600 ms only for ambiguous queries). Note an explicit evidence gap regarding your wake-word and CTC notes: no independent peer-reviewed benchmark in the surveyed literature verifies RepCNN's cited 10x speedup over BC-ResNet or evaluates MatchboxNet-CTC specifically on Filipino-English accents; those figures remain internal hypotheses to be confirmed on your own recorded holdout set.

For edge speech synthesis, independent CPU deployment guides corroborate your caution regarding **Kokoro-82M** (325.5 MB FP32, 92.4 MB INT8): despite strong synthesis quality, no verified real-time Raspberry Pi 4 benchmark exists. **Piper TTS** (VITS on ONNX Runtime) remains Home Assistant's default local synthesizer, with reported Raspberry Pi 4 performance around 0.35–0.81 RTF depending on thread count (~0.36 RTF on medium voices in Sherpa-ONNX/KMP reports, and ~300 ms latency in developer Pi 4 assistants). Choosing **low-tier 16 kHz Piper voices** (28–70 MB) halves audio sample generation compared to medium 22.05 kHz voices (~63 MB), creating substantial CPU headroom on Raspberry Pi 4. For lower first-audio latency on CPU without abandoning ONNX, the **Piper-Plus** repository reports that its unified MB-iSTFT-VITS2 decoder reduces median single-inference P50 latency to 27 ms (23% faster than original Piper at 35 ms and 49% faster than baseline sherpa-onnx VITS at 53 ms) in a 38 MB model.

### The Edge Physics Floor: Turn-Taking Latency, Energy, Thermals, and Storage I/O

Your measured ~1.0–1.1 s pre-thinking floor (0.64 s end-of-turn hold + 0.46 s offline Parakeet pass) aligns with HCI turn-taking research and industry voice-cascade latency budgets. Psycholinguistic research shows natural human turn-taking gaps sit at **200–500 ms** because listeners predict turn boundaries and overlap ~600 ms of cognitive planning while the speaker finishes. Sequential AI cascades cannot begin STT, LLM, and TTS until silence is confirmed: Twilio's engineering target for a straightforward cascade is **1,115 ms** mouth-to-ear (350 ms STT + 375 ms LLM first token + 100 ms TTS first byte plus transport and buffering), and local cascades with a 500–600 ms endpointing wait typically land near **1,250 ms**. This confirms that running your Stage-0 acoustic CTC fast path (13.6 ms network compute on a single thread) in parallel is the most viable cascade-compatible lever to break the 500 ms conversational barrier for fixed smart-home commands without switching to an end-to-end speech-to-speech architecture.

Beyond neural compute, three systems-level Raspberry Pi 4 constraints routinely undermine sustained 24/7 latency if overlooked:

1. **Thermal throttling and energy scaling:** Hardware USB power-meter evaluations on Raspberry Pi 4B demonstrate that normal-mode CPU ASR draws ~1.6 W and that **energy consumption grows exponentially as Word Error Rate improves linearly**, while Jeff Geerling's sustained Pi 4 benchmarks show that running continuous workloads without active airflow induces thermal throttling that degrades speed by **2–3x**. Any continuous soak test or production deployment on Pi 4/5 requires active fan cooling to prevent silent RTF drift.
2. **The Raspberry Pi 4 A2 microSD bottleneck:** Because the Raspberry Pi 4 SD controller and standard Linux drivers do not exploit A2 command queuing, A2 cards (rated up to 2,000 random 4K write IOPS) perform no better—and sometimes worse—than cheaper A1 cards. Raspberry Pi 4 Diagnostics commonly measure ~904–955 random 4K write IOPS on solid A1/EVO Plus cards (with some A1 cards failing below 300 IOPS), whereas non-SD storage benchmarks in the same diagnostics thread reach **20,289 write IOPS**. Workloads with sustained random writes should use USB 3.0 external storage or verified A1 cards (SanDisk Ultra A1 or Samsung EVO Plus).
3. **SQLite concurrency and FTS5 tuning:** Unoptimized SQLite defaults (`journal_mode = DELETE`, `synchronous = FULL`, 8 MB page cache) serialize readers behind writers and `fsync` every commit, stalling microSD I/O. Official SQLite documentation and engineering benchmarks confirm that switching to `WAL` plus `synchronous = NORMAL` allows concurrent reads during writes and accelerates commits by **2–10x** (at the minor cost of potentially losing the last uncheckpointed transaction on sudden power loss, without database corruption). For local `doc_chat` lexical search over hyphenated identifiers, part numbers, or error codes, configure FTS5 explicitly:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -64000;      -- ~64 MB page cache on Pi
PRAGMA mmap_size = 268435456;    -- 256 MB memory-mapped reads

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text,
  content='chunks',
  content_rowid='id',
  tokenize="unicode61 remove_diacritics 0 tokenchars '-_'"
);

-- Disable incremental merge during bulk ingest, then restore (default 4, max 16)
INSERT INTO chunks_fts(chunks_fts, rank) VALUES('automerge', 0);
```

*(Note: External content FTS5 tables do not support `REPLACE` conflict handling and require triggers or an explicit `'rebuild'` command to remain synchronized with the base table.)*

---

## Large-Scale and Hybrid RAG Validation: Ingestion, Retrieval, and Graph Economics

Although your notebook centers on the voice cascade and multi-turn agent harness, your `doc_chat` and `web_research` pathways intersect directly with the primary failure modes identified in 2024–2026 enterprise RAG literature. The consistent empirical finding across benchmarks is that **RAG pipelines rarely plateau because of generator capacity; they fail upstream due to parsing corruption, uncalibrated hybrid scoring, and mismatches between query type and index structure.**

### Document Parsing, OCR Ceilings, and the Numeric Table Failure Mode

Enterprise benchmarks expose severe breakdowns when standard text-chunking RAG encounters complex PDFs, multi-column layouts, and numerical filings. On **FinanceBench** (SEC filings and earnings calls), Islam et al. (2023) reported that GPT-4-Turbo paired with standard retrieval incorrectly answered or refused **81% of questions**. Two 2026 studies isolate the root mechanisms:

- **OCR layout blind spots:** **InduOCRBench** (ACL 2026 Industry) evaluated 10 OCR models across industrial document layouts (multi-column, cross-page tables, visually styled layouts) and identified an "OCR Blind Spot" regime (`VisualStyle`) where **82.9% character-level OCR accuracy yields only 53.0% downstream RAG accuracy** due to corrupted reading order and lost format semantics. Consistent with this, a vendor benchmark by Mixedbread reported that leading OCR tools trail ground-truth text by ~4.5% NDCG@5 (0.74 vs. 0.773) and incur a **25.8% drop in downstream answer correctness** (0.677 vs. 0.913), while Mixedbread's multimodal visual vector store outperformed ground-truth text by ~12% NDCG@5 (0.865) by preserving chart and table layout directly.
- **Numerical table degradation under RAG:** **FinTradeBench** (2026) evaluated large, mid-sized, and small LLMs—including **Gemini 2.5 Flash** and **Gemini 2.5 Flash-Lite**—across SEC prose-based fundamental extraction versus OHLCV technical-trading numeric calculations. Retrieval delivered strong gains on fundamental prose questions (R1-Distill-Qwen 32B **+37.0%**, Gemini 2.5 Flash **+23.8%**, DeepSeek-R1 **+23.6%**, GPT-5-mini **+13.1%**), yet **degraded numerical trading accuracy across nearly every evaluated model**—including **-19.7% on Gemini 2.5 Flash-Lite**. Even when retrieval located the exact price tables, generators failed to compute multi-step indicators reliably from unrolled text chunks—confirming that numerical table queries should be routed to **Program-of-Thought / sandboxed code execution**, which **FinAgent-RAG** (2026) showed eliminates **88.0% of arithmetic errors** across FinQA, ConvFinQA, and TAT-QA.

#### Benchmark Table 2A: FinTradeBench Relative RAG Delta vs. No-RAG (%)

*Relative accuracy change `((RAG - No-RAG) / No-RAG) x 100%` on SEC Fundamental prose vs. OHLCV Trading numeric tables.*

| Model | Fundamental (Prose): No-RAG -> RAG | Fundamental Delta | Trading (Numeric): No-RAG -> RAG | Trading Delta |
| :--- | :--- | :--- | :--- | :--- |
| R1-Distill-Qwen 32B | 31.7% -> 43.5% | +37.0% | 21.6% -> 22.0% | +2.2% |
| Gemini 2.5 Flash | 31.0% -> 38.4% | +23.8% | 24.1% -> 21.6% | -10.3% |
| DeepSeek-R1 | 34.0% -> 42.0% | +23.6% | 24.8% -> 24.8% | 0.0% |
| GPT-5-mini | 37.1% -> 42.0% | +13.1% | 27.8% -> 23.2% | -16.4% |
| **Gemini 2.5 Flash-Lite** | **34.4% -> 34.4%** | **0.0%** | **26.4% -> 21.2%** | **-19.7%** |
| LLaMA 3.3 Instruct 70B | 40.1% -> 36.9% | -7.9% | 25.0% -> 20.5% | -18.0% |

To control ingestion quality without wasting compute on born-digital pages, modern parsing pipelines apply **per-page OCR triage** based on image-area coverage:

$$r(p) = \frac{A_{\text{img}}(p)}{A(p)}, \quad \text{route}(p) = \begin{cases} \text{OCR} & \text{if } r(p) > \tau_{\text{cov}} \\ \text{native parser} & \text{otherwise} \end{cases}$$

Using a default coverage threshold of $\tau_{\text{cov}} = 0.8$ rasterizes only scanned pages while routing native pages to PyMuPDF. When OCR is required, layout preservation matters more than raw character count: in an engineering comparison on a 1974 scanned PDF, **Docling** extracted 5,423 characters with 11 table-of-contents entries and 4 figure regions structured into 105 paragraphs (at 2.3x compute), whereas **EasyOCR** produced 4,952 characters fragmented across 346 unstructured bounding boxes. Security-sensitive ingestion pipelines must also account for **single-parser divergence**: PDF forensics research shows AcroForms maintain separate `/V` (machine value) and `/AP` (human-visible appearance) stores, and incremental PDF updates can hide form fields or text that MuPDF ignores while Poppler extracts, creating a stealth RAG poisoning vector.

### Chunking, Header Prepending, and Why Unnormalized BM25 Thresholds Break

Table fragmentation destroys retrieval precision before embedding even runs. In a vendor-reported FinanceBench test, Ragie found that prepending table headers to every chunk and preventing mid-row splits boosted performance by **58%** (single store) and **137%** (shared store) at `top_k = 128`. Independent academic studies confirm the same directional benefit: prepending hierarchical heading paths raised recall@5 by **+6.9 percentage points** on university regulation documents; Anthropic's **Contextual Retrieval** (prepending a 50–100 token document-context header using cached LLM calls at ~80% cost savings) cut retrieval failure rates by up to **67%**; and a 2025 FinanceBench preprint using RAGChecker showed contextual chunking plus a custom metadata-aware reranker reached **43.2 F1** (+31% relative over Naive RAG's 32.9 F1, nearly matching commercial Cohere reranking at 44.4 F1).

Crucially, that FinanceBench ablation exposed a recurring engineering pitfall: **hybrid retrieval on standard chunks without a reranking step scored 30.4 F1—worse than Naive RAG (32.9 F1)**—because merging sparse and dense lists injects unranked noise without cross-encoder or metadata reranking. Applying a static `min_score` cutoff to raw BM25 outputs is similarly brittle: because BM25 sums saturating term frequencies weighted by unbounded IDF values over variable query lengths, raw scores are not comparable across queries. Elasticsearch engineering guidance notes that `minmax` normalization pins the worst candidate to 0 and can cause valid documents with weak lexical overlap to be dropped by a hybrid `min_score` cutoff. Three validated alternatives replace fixed BM25 thresholds:

1. **Reciprocal Rank Fusion (RRF):** Combines BM25 and dense candidates strictly by rank position, applying `min_score` filtering only to the normalized semantic retriever.
2. **Maximum-Normalized Lexical Scoring (BMX):** Normalizes lexical scores into $[0, 1]$ by dividing by the theoretical query maximum $\text{score}(D, Q)_{\max} = m \cdot [\log(1 + \frac{n - 0.5}{1.5}) + 1.0]$, outperforming standard BM25 on 11 of 15 BEIR datasets while supporting stable thresholds.
3. **Query Performance Prediction (QPP):** Evaluates post-retrieval score concentration and variance across top-k results—using metrics like **Normalized Query Commitment (NQC)** and **Weighted Information Gain (WIG)**—to decide when to abstain or escalate to web search.

### GraphRAG vs. LightRAG vs. LiteRAG vs. Temporal RAG Economics

When scaling `doc_chat` or enterprise knowledge bases, teams frequently consider **Microsoft GraphRAG**. Recent 2025–2026 cost and temporal benchmarks clarify where graph indexing earns its overhead—and where it fails outright.

| Architecture | Core Mechanism | Measured Cost & Latency Profile | Best Workload & Verified Win Rate | Documented Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla / Hybrid RAG + Reranker** | Sparse BM25 + dense vectors + RRF + reranker | Lowest ingest; <100 ms retrieval; flat per-query cost | Local factual lookups; 56.33% on TimeQA (beating static GraphRAG by +16–46 pp) | Misses corpus-wide themes and multi-hop cross-document synthesis |
| **RAPTOR** | Recursive clustering and summary tree | Moderate one-time tree build; multi-level vector lookup | Multi-scale abstraction (+20 pp absolute lift on QuALITY with GPT-4) | Static tree requires re-clustering on frequent corpus updates |
| **Microsoft GraphRAG (Global / DRIFT)** | Entity graph + hierarchical community reports + map-reduce | Very high: **162.66 s / 643k tokens** per Global query; DRIFT >142 s / 6.4M tokens; AWS Labs reports gleaning = 47% ($173.6 of $371.64) of build cost | Corpus-wide thematic sensemaking (70–80% win rate vs. naive RAG) | Collapses on temporal QA (**10.10%** TimeQA, **20.97%** ComplexTR) and granular lookups; expensive full rebuilds |
| **LightRAG** | Incremental entity/relation KV pairs inside vector store; dual-level search | Secondary guides cite ~90% fewer API calls and single-call retrieval; runs on local Ollama LLMs | Dynamic corpora needing incremental updates (82.54% win rate vs. naive RAG on UltraDomain Legal) | Can be slower than GraphRAG on small corpora due to longer per-call outputs; trails vanilla/temporal RAG on TimeQA (36.36%) |
| **LiteRAG** | Reuses graph index with query-conditioned algorithmic traversal (no retrieval-time LLM loop) | **1.42 s latency, 2,291 tokens, $0.0003/query** (>99% cost cut, >100x faster than GraphRAG Global) | Budget-constrained graph RAG (~2,000-token cap with 0.0% quality drop on DistComp) | Does not reduce upfront graph indexing cost |
| **IA-RAG (Interval-Aware)** | Interval Event Units + Allen's Interval Algebra | Structured interval graph traversal | Evolving/conflicting temporal facts (**61.72%** TimeQA, **80.21%** TempReason, **65.95%** ComplexTR) | Extraction overhead unnecessary for static corpora |
| **Self-Route (Hybrid Router)** | Per-query classifier routing between RAG and Long-Context LLM | Cuts **35–60% of tokens** at comparable quality | Mixed workloads (RAG and long-context agree on >60% of queries) | Requires calibrated router threshold to avoid misrouting global synthesis |

#### Benchmark Table 2B: Temporal QA Accuracy & Recall (%) on TimeQA and ComplexTR

| Method | TimeQA Accuracy | TimeQA Recall | ComplexTR Accuracy | ComplexTR Recall |
| :--- | :--- | :--- | :--- | :--- |
| **IA-RAG (Interval-Aware)** | **61.72%** | **69.25%** | **65.95%** | **77.42%** |
| DyG-RAG | 58.78% | 67.02% | 55.62% | 69.88% |
| Vanilla RAG | 56.33% | 64.98% | 55.01% | 68.68% |
| GraphRAG-Local | 40.26% | 46.28% | 43.16% | 54.29% |
| LightRAG-Hybrid | 36.36% | 43.06% | 42.68% | 53.59% |
| GraphRAG-Global | 10.10% | 13.83% | 20.97% | 30.81% |

Two design implications follow directly. First, if you add graph memory to a large or evolving `doc_chat` corpus, skip Microsoft GraphRAG in favor of **LightRAG** (for incremental updates on commodity hardware) or **LiteRAG's algorithmic traversal**—and if GraphRAG is ever required for static global sensemaking, AWS Labs ablations show capping gleaning at `max_rounds: 1` eliminates roughly two-thirds of gleaning cost while capping global evidence at 4,000 tokens saves 41% per query with no consistent quality drop. Second, because 23% of factual queries in Meta's CRAG are slow-changing and another 23% are fast-changing or real-time, persistent document stores must bind facts to explicit validity intervals or versioned write-time ledgers (as in **IA-RAG** and **EvoWiki**) rather than unversioned chunks.

---

## Agentic Reliability: Abstention, Verification Gates, and Multi-Turn Context

### Why Intrinsic Self-Correction Fails—and Why Naive RAG Cuts Truthfulness

Your RQ1 design—rejecting unconstrained Haiku reflection loops in favor of a deterministic `SOURCES.json` and `index.html` completion gate—is strongly backed by peer-reviewed reasoning research. In **Huang et al. (ICLR 2024), "Large Language Models Cannot Self-Correct Reasoning Yet,"** researchers removed the hidden oracle labels used in earlier self-correction papers (such as Reflexion and RCI) and evaluated *intrinsic self-correction* (asking an LLM to critique and revise its own answer without external feedback). Across every evaluated model and dataset—GPT-3.5-Turbo, GPT-4, GPT-4-Turbo, and Llama-2-70b-chat on GSM8K, CommonSenseQA, and HotpotQA—**intrinsic self-correction consistently lowered accuracy**. Models flipped correct answers to wrong answers more often than they corrected mistakes, while multiplying LLM calls 3x to 5x. Reliable correction requires external, deterministic feedback such as compilers, schema checks, unit tests, or tool outputs.

#### Benchmark Table 3A: Intrinsic Self-Correction Accuracy (%) Without External Verifiers (Huang et al., ICLR 2024)

| Model & Benchmark | Standard Prompting (1 call) | Self-Correct Round 1 (3 calls) | Self-Correct Round 2 (5 calls) | Net Change (R2 vs. Standard) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3.5-Turbo — CommonSenseQA | 75.8% | 38.1% | 41.8% | -34.0 pp |
| GPT-4 — GSM8K | 95.5% | 91.5% | 89.0% | -6.5 pp |
| GPT-4 — HotpotQA | 49.0% | 49.0% | 43.0% | -6.0 pp |
| Llama-2-70B — GSM8K | 62.0% | 43.5% | 36.5% | -25.5 pp |
| Llama-2-70B — CommonSenseQA | 64.0% | 37.5% | 36.5% | -27.5 pp |

In retrieval-augmented generation, **Meta's CRAG benchmark** (NeurIPS 2024; 4,409 QA pairs across 5 domains) exposes an analogous trap: accuracy-only dashboards mask costly overconfidence. CRAG scores correct answers as +1, abstentions as 0, and hallucinations as -1. Under that truthfulness metric, retrieving 50 web pages plus a knowledge graph lifted GPT-4 Turbo's raw accuracy from 33.5% to 43.6%, yet **lowered its net truthfulness from 20.0 to 13.4** because hallucinations more than doubled (13.5% to 30.1%) as cautious abstentions fell (53.0% to 26.3%).

#### Benchmark Table 3B: Meta CRAG GPT-4 Turbo Across Retrieval Settings (Auto-Eval)

| Retrieval Setting | Raw Accuracy (%) | Hallucination Rate (%) | Missing / Abstention (%) | Net Truthfulness Score (pts) |
| :--- | :--- | :--- | :--- | :--- |
| LLM Only (No Retrieval) | 33.5% | 13.5% | 53.0% | **20.0** |
| Task 1: 5 Pages | 35.9% | 28.2% | 35.9% | 7.7 |
| Task 2: 5 Pages + KG | 41.3% | 25.1% | 33.6% | 16.2 |
| Task 3: 50 Pages + KG | 43.6% | 30.1% | 26.3% | 13.4 |

Commercial search-augmented assistants show a steep latency-truthfulness trade-off as well: under human grading, Copilot Pro achieved 50.6 truthfulness at 11.6 s latency, Gemini Advanced reached 49.3 at 5.2 s, ChatGPT Plus reached 41.5 at 6.2 s, and Perplexity.ai reached 34.9 at 4.6 s—whereas KDD Cup 2024 first-place winner `db3` won all three tasks (28.4 / 42.7 / 47.8) by prioritizing selective abstention and query routing.

Subsequent 2026 evaluations indicate that **behavioral judgment under uncertainty** remains the central bottleneck for enterprise agents. **EnterpriseRAG** (983 expert-validated queries across 13 LLMs) uncovered an orchestration gap of up to **57 percentage points** between Loose per-constraint satisfaction (83.8%) and Strict holistic compliance (26.8%), with peak negative-rejection accuracy at **42.7%** and conflict recognition **under 45%**. For pipelines whose outputs trigger downstream tool side effects (such as `delegate` or `publish_site`), even CRAG's -1 hallucination penalty may be too lenient. Adding **CRAG-style 1/0/-1 truthfulness scoring** to `scripts/eval_chain.py`—and explicitly rewarding calibrated abstention, as demonstrated by the GRPO-based **TruthRL** approach (which improved truthfulness to 71.9 vs. 66.7 prompting while reducing hallucination from 16.3% to 12.7% in OpenReview evaluations)—helps ensure your agent abstains rather than fabricating claims when `web_research` returns ambiguous or conflicting snippets.

### The Hallucination Detector Operating-Point Trap

Inserting a small NLI or grounding model—such as **MiniCheck** (770M Flan-T5-Large), **AlignScore**, or **LettuceDetect** (ModernBERT-based)—as an automatic blocker between `web_research` and spoken replies is a common proposal, yet 2026 empirical audits demonstrate why **ranking metrics (AUROC) fail to predict production gating viability**. In an independent evaluation on RAGTruth QA, tuning detectors to catch **95% of hallucinations** yielded a best-case false-positive rate of **0.65**—meaning in a workload with a 15% defect rate, the gate flags **69.5% of all traffic**. Even the highest-AUROC trained span detector (0.817 AUROC) flagged **99% of correct answers** at 95% recall, and a Ragas-style judge exhibited a 1.00 false-positive rate across all five datasets tested.

This fragility stems partly from stylistic shortcut overfitting: in the same audit, LettuceDetect retained **0.635 AUROC with zero grounding evidence provided** (43% of its above-chance signal came purely from phrasing and length fingerprints) and inverted to **0.415 AUROC (worse than random)** out-of-distribution on FinanceBench, while single-digit numeric substitutions routinely bypassed trained fact-checkers. Furthermore, a 2026 TU Wien thesis on latency-tiered detection found that **cascade routing** (gating a larger judge behind a fast detector) degraded F1 by **-0.026 to -0.042** due to bimodal, uncalibrated score mixing. Where lightweight verification did succeed was **parallel score blending**: combining LettuceDetect-Large with a Qwen 2.5 14B Grounding Probe ($\alpha = 0.45$) achieved **0.924 AUROC and 0.820 F1 at ~94 ms** parallel latency, while a Qwen 2.5 3B variant reached **0.915 AUROC at ~49 ms**. Consequently, avoid deploying MiniCheck or NLI classifiers as silent binary blockers on your voice fast path; rely on deterministic checks (URL/entity presence in `SOURCES.json` and exact numeric verification against snippets) for hard blocking, and reserve grounding verifiers for offline regression audits or soft review triggers.

### Context Position, Prefix Caching, and Multi-Turn Carryover

Your RQ6 (deterministic `FocusState` + selective pronoun escalation) and RQ9 (keeping the ~3.4k-token static system prompt immutable at the top while appending `[ACTIVE TASK]` to the end of the user message) align closely with empirical long-context and conversational QA research. In **Liu et al., "Lost in the Middle,"** shifting a single answer-bearing passage across 20 retrieved documents (~4K tokens) caused **GPT-3.5-Turbo** accuracy to fall **22.0 percentage points**—from **75.8%** at Position 1 to **53.8%** at Position 10, below its **56.1%** closed-book baseline with zero documents—while `GPT-3.5-Turbo (16K)` displayed a nearly identical U-shaped curve. Subsequent 2025–2026 evaluations (**NoLiMa**, **RULER**, and Chroma's 18-model **Context Rot** study) indicate that recall degrades with context fill and that even a single near-miss distractor reduces accuracy. Keeping your ~3.4k-token Gemini prefix strictly static clears the 1,024-token minimum for Gemini Flash implicit caching (up to a 90% input-token discount), while placing dynamic `[ACTIVE TASK]` headers and capped `web_research` extracts (<=2,000 characters) at the recency tail mitigates mid-context attention loss.

Conversational search benchmarks similarly support **selective follow-up resolution** over either ignoring history or rewriting every utterance. On QReCC, **ConvGQR** (Mo et al., ACL 2023) raised retrieval MRR from **10.2** (raw follow-up query) to **42.0**, surpassing human-written rewrites (38.4), and an incomplete follow-up resolution plug-in for Siri delivered a **131.57% improvement** on elliptical queries. Conversely, unconditional rewriting carries documented downside: a 2026 evaluation found prompt-only query rewriting gained +5.1% nDCG@10 on TREC-COVID but **dropped -9.0% on FiQA**, while **TopiOCQA** benchmarks introduced the **over-carry rate (OC)** to measure how often multi-turn models erroneously inject stale prior entities across topic pivots. Keeping your 350M local router locked at `HISTORY_PAIRS = 1` (preventing the ~0.3 s/pair CPU prefill cost and `llama.cpp` prefix-cache invalidation), escalating only Distance-2+ or unresolved pronoun utterances to Gemini with a 1-line `FocusState` hint, and expiring chain state after 3 unrelated turns directly addresses both failure modes. Strengthen `tests/eval_followups.jsonl` by scoring TopiOCQA's **over-carry rate (OC)** explicitly on topic-pivot traps and preserving the raw unrewritten query as a parallel retrieval branch in `doc_chat`.

---

## Security Hardening: Indirect Prompt Injection, Sandboxes, and Archive Ingestion

Your notebook rightly treats indirect prompt injection (IPI) as the top threat when `web_research` or `delegate` consumes untrusted web pages, and wisely keeps GitHub PATs outside the `bwrap` sandbox. However, 2024–2026 agentic security benchmarks show that `<untrusted_search_data>` XML delimiters and `--share-net` inside Bubblewrap leave exploitable gaps.

### Why Delimiting Fails and Which IPI Defenses Actually Hold

In **Hines et al. (2024), "Defending Against Indirect Prompt Injection Attacks with Spotlighting,"** Microsoft researchers compared **delimiting**, **datamarking**, and **encoding** on BIPIA. Naive **delimiting** (wrapping external content in `<<...>>` or XML tags) only lowered GPT-3.5-Turbo Attack Success Rate (ASR) from roughly **60% to 30%** and breaks down as soon as an adversary includes the closing delimiter (`</untrusted_search_data>`) inside a web page. **Datamarking**, by contrast, interleaves a platform-applied sentinel character (such as `^` replacing whitespace or inserted at tokenizer boundaries) continuously across every word of the untrusted payload, giving the model an unforgeable token-level provenance signal. On GPT-3.5-Turbo summarization, datamarking cut ASR from ~50% to **3.1%** (and from 40% to **0%** on `text-davinci-003`; survey aggregates cite >50% down to <2%) with negligible degradation on SQuAD, BoolQ, Word-in-Context, and IMDB. **Encoding** (such as Base64) lowers ASR even further on frontier models, though datamarking imposes far less cognitive overhead on smaller models like Gemini Flash-Lite or Haiku.

#### Benchmark Table 4: Indirect Prompt Injection Defense Comparison (ASR %, Lower Is Safer)

*Target agentic risk ceiling: `< 5%` ASR.*

| Defense Paradigm | Reported ASR | Baseline / Benchmark | Security Tier | Operational Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| External Chat-Safety Classifiers | 43.8–44.0% | INFUSE high-ASR stress subset | Weak (`> 10%`) | Transfers poorly to environment-embedded payloads |
| Naive Delimiting (`<<...>>` / XML) | 30.0% | Spotlighting (from ~60%, GPT-3.5-Turbo) | Weak (`> 10%`) | Trivially escaped by injecting closing tag |
| Other System Patterns (3 of 6) | ~20.0% | Large-Scale Red-Teaming (13k+ attacks) | Weak (`> 10%`) | Residual exposure across non-isolated designs |
| Prompt-Level Hardening | 14.3% | INFUSE mean ASR-L (from 58.2%) | Weak (`> 10%`) | Uneven, model-dependent utility drops |
| **Spotlighting Datamarking** | **3.1%** | Spotlighting (from ~50%, GPT-3.5-Turbo; 0% davinci-003) | **Validated (`< 5%`)** | Negligible utility loss on SQuAD / SuperGLUE / IMDB |
| **Structural Isolation** (Plan/Code-Then-Execute, Dual-LLM) | **1–5%** | Large-Scale Red-Teaming (13k+ attacks) | **Validated (`< 5%`)** | Strongest system-level defense; highest utility trade-off |
| **Representation Defense — Odile** | **0.01%** | COLM 2026 (from 14.04%, Llama-3.3-70B; 0.2% on 8B) | **Validated (`< 5%`)** | Benign utility preserved (59.90% -> 59.80%); requires open weights |

Large-scale 2025–2026 agentic benchmarks reinforce why regex filters and external safety classifiers fail under adaptive attacks. Static regex filters are routinely bypassed through Base64/Unicode encodings, token fragmentation (zero-width characters or concatenation), and cross-language translation. On **INFUSE** (41,824 agent trajectories across 40 long-horizon tasks), mean attack success reached **19.6%**, explicit refusal occurred in **<=1.0%** of cases, and external chat-safety classifiers suffered **43.8–44.0% ASR** on stress subsets; prompt-level hardening reduced stress ASR from 58.2% to **14.3%**, still too risky for agents with shell or publishing tools.

By contrast, a 2026 large-scale review of 13,000+ unique public red-teaming attack strings found that three system-level patterns (Beurer-Kellner et al., 2025)—**Plan-Then-Execute**, **Code-Then-Execute**, and **Dual-LLM**—hold residual ASR to **1–5%** versus ~20% for non-structural patterns, though reviewers emphasize that these stricter patterns also carry the largest benign utility reduction. Your separation between Gemini (front planner), Haiku (sandboxed builder), and a host-side validated `publish_site` tool already approximates a **Dual-LLM / Plan-Then-Execute** boundary—provided untrusted `web_research` text cannot dynamically alter tool permissions. For large-scale deployments fine-tuning open models, **Odile** (COLM 2026) separates benign and attacked intent-formation trajectories to lower ASR from **14.04% to 0.01%** on Llama-3.3-70B (benign utility 59.90% -> 59.80%) and from **16.0% to 0.2%** on Llama-3.1-8B (26.0% unchanged), outperforming Meta-SecAlign (4.1% ASR) and Prompt Guard 2 (7.9% ASR). In addition, a March 2026 non-peer-reviewed preprint (**RUI**, tested on 5,000 BIPIA samples) reports adaptive injection ASR under **8.1%** with **>88.8%** benign performance preservation.

### Closing Sandbox Network and Archive Ingestion Gaps

Two systems-level runtime vulnerabilities require direct remediation at the edge/hybrid boundary:

1. **Bubblewrap loopback exposure:** Running Bubblewrap with `--share-net` allows prompt-injected curl or shell code inside the Haiku sandbox to probe `127.0.0.1` (your unauthenticated local Agent Server and `llama.cpp` router) and `169.254.x.x` (Raspberry Pi link-local services). Enforce bearer-token authentication on the Agent Server and apply an `iptables`/`nftables` UID-owner egress rule blocking `127.0.0.0/8` and `169.254.0.0/16`—or better yet, drop `--share-net` inside `bwrap` completely and route all web lookups exclusively through your host-side `web_research` MCP tool (which strips scripts and applies datamarking before returning text).
2. **Archive ingestion CVEs:** Any local or cloud ingestion service unpacking `.zip`, `.docx`, `.xlsx`, or `.tar` archives faces two active CVE classes: **Zip Slip path traversal (CWE-22)**—including nested archive and path-stripping bypasses such as `jaraco.context` (`dummy_dir/../../etc/passwd`)—and **decompression bombs (CWE-400 / CWE-770)**, where crafted gzip/zip payloads achieve ~1000:1 compression ratios or spoof ZIP `file_size` headers to exhaust host heap memory. Hardening requires canonicalizing every extraction path with an `os.path.realpath` prefix check, rejecting symlinks, and enforcing **streaming chunk-read byte limits** on actual inflated output rather than trusting archive metadata headers.

---

## Integrated Hardening Blueprint: Edge and Large-Scale Deployment Roadmap

Synthesizing verified edge hardware measurements, enterprise RAG benchmarks, and agentic security studies yields a unified two-tier architecture that preserves your lean, framework-free design while systematically closing its highest-probability failure modes.

```text
====================================================================================
                      HARDENED EDGE-TO-CLOUD ARCHITECTURE
====================================================================================

[ TIER 1: EDGE NODE (Raspberry Pi 4/5 + Active Cooling + A1 / USB 3.0 Storage) ]
------------------------------------------------------------------------------------
  16 kHz Audio Stream
|
         v
  +-----------------------------+

| DS-CNN Wake-Word Firewall | (Always-on, low CPU)
  +--------------+--------------+

| (Confirmed Trigger)
        +--------+----------------------------+

| |
        v (Fixed Smart-Home Fast Path)        v (General / Open Turns)
  +---------------------------+     +----------------------------------------+

| Stage-0 Acoustic CTC Trie | | Local STT (Parakeet-TDT / Moonshine) |
| (13.6 ms compute, 1 core) | | (All 4 cores sequential; Kroko paused) |
  +------------+--------------+     +-------------------+--------------------+

| |
               v                                        v
      [ Execute Local IoT ]             +------------------------------------+

| 350M Granite Router (GBNF Grammar) |
| (HISTORY_PAIRS = 1, 0.24s median) |
                                        +--------+------------------+--------+

| |
                     (Simple Local Tool / Lookup)| |(Open-ended /
                                                 v | Distance-2+ Pronoun)
                                        +----------------+ |

| WAL SQLite FTS5| |
| + Local Tools | |
                                        +--------+-------+ |

| |
                                                 v |
                                        [ Piper / Kitten TTS ] |
|
====================================================================|===============
[ TIER 2: CLOUD & HYBRID AGENT TIER (Isolated Execution) ] |
--------------------------------------------------------------------|---------------
                                                                    v
  +-------------------------------------------------------------------------------+

| Front Planner (Gemini Flash-Lite) |
| * Immutable 3.4k-token static prefix (>=1,024 tokens -> 90% implicit cache) |
| * Dynamic [ACTIVE TASK] + FocusState appended at user-turn tail |
  +---------------------------------------+---------------------------------------+
|
                 +------------------------+------------------------+

| |
                 v (Untrusted Web / RAG Lookup)                    v (Numeric Tables)
  +-----------------------------------------+       +-----------------------------+

| Host-Side web_research / RAG Retriever | | Sandboxed Python Execution |
| * Strip scripts & cap <=2,000 chars | | (Program-of-Thought; cuts |
| * Apply Spotlighting Datamarking (^) | | arithmetic errors by 88%) |
| * Zip Slip + streaming byte limits | +-----------------------------+
  +--------------------+--------------------+

| (Datamarked Data Only — No Credentials)
                       v
  +-------------------------------------------------------------------------------+

| Isolated Builder Sandbox (Haiku Delegate inside bwrap) |
| * NO --share-net (blocks 127.0.0.1 & 169.254.x.x) |
| * NO GitHub / HF tokens in filesystem or env |
  +---------------------------------------+---------------------------------------+
|
                                          v
  +-------------------------------------------------------------------------------+

| Deterministic Completion Gate (Zero LLM Self-Correction) |
| * Verify index.html & SOURCES.json exist, non-empty, URL/fact grounded |
  +---------------------------------------+---------------------------------------+
|
                                          v
  +-------------------------------------------------------------------------------+

| Host-Side publish_site MCP Tool (Outside bwrap) |
| * Validate <=25 files, <=2 MB, no symlinks, no secrets, auto-.nojekyll |
| * Explicit Spoken Confirmation Gate -> Push via host-only GIT_ASKPASS |
  +-------------------------------------------------------------------------------+
```

### Prioritized Action Matrix by Deployment Tier

Execute the following upgrades in priority order across your edge and hybrid/cloud tiers to close the validated reliability and security gaps without introducing orchestration framework bloat:

| Priority | Deployment Tier | Concrete Engineering Upgrade | Verified Benchmark / Research Basis | Verification Metric & Pass Bar |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Large / Hybrid (Security)** | Upgrade `<untrusted_search_data>` XML tags to **Spotlighting datamarking** (sentinel token interleaved across retrieved web/doc spans) and drop `--share-net` inside `bwrap` (routing web fetches strictly via host `web_research` MCP). | Spotlighting cuts ASR from ~30% (delimiting) to **3.1%**; Dual-LLM / Plan-Then-Execute holds residual ASR to **1–5%**; eliminates `127.0.0.1` / `169.254.x.x` sandbox escape. | Run 20 BIPIA/InjecAgent-style injected pages (including closing-tag escapes and `curl 127.0.0.1` probes); require **0% sandbox loopback access** and **<5% tool hijack ASR**. |
| **2** | **Edge (RPi 4/5 & Laptop)** | Allocate all **4 physical cores sequentially** (pause Kroko during Parakeet finalize; `threads=4` on Parakeet, `llama.cpp`, and Kitten/Piper), test **beam=2** on Parakeet, and add active cooling + USB 3.0 / A1 storage with SQLite `WAL` + `synchronous=NORMAL`. | Unlocks measured **-95 ms** on Parakeet (423 -> 328 ms); prevents **2–3x** thermal throttling; WAL + NORMAL gives **2–10x** faster local commits. | Confirm local-router median drops from **1.53 s to <=1.35 s** with hard-set WER <=4.4%, and 4-hour Pi soak shows <5% RTF drift under active fan cooling. |
| **3** | **Large / Hybrid (Evaluation)** | Adopt **Meta CRAG's 1/0/-1 truthfulness score** inside `scripts/eval_chain.py` alongside $\text{pass}^4$, and add an **over-carry rate (OC)** check to `tests/eval_followups.jsonl`. | Plain accuracy hides hallucination surges (CRAG GPT-4 Turbo: 43.6% accuracy but 13.4 truthfulness); TopiOCQA shows multi-turn models over-carry stale entities on topic pivots. | Require $\text{pass}^4 = 100\%$ on 4-trial chain replay, net truthfulness strictly positive on unanswerable probes (explicit abstention), and OC <5% on topic-pivot traps. |
| **4** | **Large / Hybrid (RAG & Ingestion)** | Implement **per-page OCR triage** ($r(p) > 0.8 \to$ Docling; native $\to$ PyMuPDF), **header-prepended table chunking**, **RRF + reranking** (never un-reranked hybrid or raw BM25 `min_score`), and **SQLite FTS5 `tokenchars '-_'`** + **Zip Slip / streaming byte limits**. | Un-reranked hybrid underperforms Naive RAG (30.4 vs. 32.9 F1); header-prepended table chunks lift FinanceBench +58% to +137% (vendor report); protects against CWE-22 and 1000:1 bombs. | Measure recall@5 delta on mixed scanned/tabular PDFs (target >=+6 pp) and verify unit tests block `../` Zip Slip paths and >10 MB decompressed streams. |
| **5** | **Large / Hybrid (Numeric & Temporal)** | Route quantitative table/calculation queries to **sandboxed Python execution** (Program-of-Thought) rather than raw RAG generation, and tag time-sensitive facts with explicit **validity intervals** (`[valid_from, valid_to]`). | RAG degrades Gemini 2.5 Flash-Lite by **-19.7%** on numeric trading data; Program-of-Thought cuts arithmetic errors by **88.0%**; IA-RAG gains **+10.33 pp** on ComplexTR. | Verify 100% deterministic calculation accuracy on numeric table queries via code execution and zero stale-fact overrides when newer timestamped sources exist. |
| **6** | **Edge (RPi 4 Standalone Path)** | If deploying fully standalone on Raspberry Pi 4 without the laptop, benchmark **Moonshine Tiny (27M)** for sub-5 s open commands alongside your **CTC grammar model** and **Piper low-tier (16 kHz)** or **Piper-Plus MB-iSTFT (38 MB)** TTS. | Moonshine Tiny cuts 1 s audio RTF from 2.65 to **0.15**; Piper low-tier halves sample generation on Pi 4; Piper-Plus MB-iSTFT reports **27 ms P50** on CPU. | Confirm end-to-end standalone Pi 4 command-to-audio latency stays under **1.0 s** for fixed commands (CTC fast path) and under **1.8 s** for open commands within 4 GB RAM. |

---

## Core Takeaway

Across both resource-constrained edge boards and cloud-scale RAG pipelines, the verified 2024–2026 evidence converges on one core engineering principle: **structural specialization outperforms raw parameter scale at every bottleneck**. On edge silicon, variable-length rotary attention (Moonshine), blank-skipping transducers (Parakeet-TDT), grammar-constrained CTC tries, and WAL-tuned local storage deliver order-of-magnitude latency and efficiency gains over general foundation models. In cloud and hybrid agent stacks, datamarked inputs, structural isolation, deterministic verification gates, interval-aware metadata, and sandboxed code execution for numeric data consistently outperform larger context windows, static community graphs, and intrinsic self-correction loops—improving factual reliability and security at a fraction of the runtime cost.

---

## Verified References

- **Edge ASR & TTS:**
  - [Turing Pi — How Voice Assistants Work (RTF, Whisper, Piper, Latency Budgets)](https://turingpi.com/how-voice-assistants-work/)
  - [LLMs.blog — Speech-to-Text Serving: Faster-Whisper vs. Moonshine vs. SenseVoice vs. NeMo Canary](https://www.llms.blog/posts/speech-to-text-serving-in-production-comparing-faster-whisper-moonshine-sensevoice-and-nemo-canary-architecture-streaming-latency-and-gpu-economics)
  - [Moonshine Paper Summary (arXiv:2410.15608)](https://0h-n0.github.io/posts/paper-moonshine-2410-15608/) & [LiteRT Moonshine-Tiny Benchmarks](https://huggingface.co/litert-community/moonshine-tiny)
  - [r/LocalLLM — Raspberry Pi 5 Sherpa-ONNX Parakeet-TDT vs. Whisper Benchmark](https://www.reddit.com/r/LocalLLM/comments/1kndn1d/benchmarking_whispers_speed_on_raspberry_pi_5_how/)
  - [Vox Changelog (RPi 4 Whisper vs. Distil-Whisper INT8)](https://github.com/mrtozner/vox/blob/main/CHANGELOG.md)
  - [Applied Sciences (2026) — Constrained Edge Cascaded Voice Pipelines](https://www.mdpi.com/2076-3417/16/16/8157)
  - [PromptQuorum — Best Local TTS on Raspberry Pi](https://www.promptquorum.com/power-local-llm/best-local-tts-raspberry-pi) & [LocalAIMaster — Best TTS Without GPU](https://localaimaster.com/blog/best-tts-without-gpu)
  - [Piper-Plus Unified MB-iSTFT-VITS2 Benchmarks](https://github.com/ayutaz/piper-plus/blob/dev/README_EN.md)
  - [Sherpa-ONNX Overview & Hardware Backends](https://ai.miraheze.org/wiki/Sherpa-ONNX)
- **Raspberry Pi 4 Thermals, Storage & SQLite:**
  - [Jeff Geerling — Raspberry Pi MicroSD Card Benchmarks & Cooling](https://www.jeffgeerling.com/blog/2019/raspberry-pi-microsd-card-performance-comparison-2019/)
  - [Performance and Efficiency Evaluation of ASR Inference on the Edge (RPi 4B Power)](https://www.researchgate.net/publication/356165601_Performance_and_Efficiency_Evaluation_of_ASR_Inference_on_the_Edge)
  - [SQLite Official FTS5 Documentation](https://sqlite.org/fts5.html) & [SQLite PRAGMA / WAL Tuning Guide](https://www.dbpro.app/blog/how-to-fix-slow-sqlite-queries)
- **RAG Parsing, Chunking, Retrieval & Graph Economics:**
  - [FinTradeBench (arXiv:2603.19225)](https://arxiv.org/html/2603.19225v3) & [FinAgent-RAG (arXiv:2605.05409)](https://arxiv.org/html/2605.05409v1)
  - [InduOCRBench (ACL 2026 Industry)](https://aclanthology.org/2026.acl-industry.60.pdf) & [Mixedbread — The Hidden OCR Ceiling](https://www.mixedbread.com/blog/the-hidden-ceiling)
  - [Metadata-Driven RAG for Financial QA on FinanceBench (arXiv:2510.24402)](https://arxiv.org/pdf/2510.24402) & [Ragie Table Chunking on FinanceBench](https://www.ragie.ai/blog/how-ragie-outperformed-the-financebench-test-part-2)
  - [BMX — Normalized Lexical Search (arXiv:2408.06643)](https://arxiv.org/html/2408.06643v2) & [Elasticsearch — Semantic Precision & Minimum Score](https://www.elastic.co/search-labs/blog/semantic-precision-minimum-score)
  - [LiteRAG vs. GraphRAG / LightRAG (arXiv:2609.10239)](https://arxiv.org/html/2609.10239v1), [AWS Labs Unified KG-RAG Evidence](https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/docs/concepts/defaults-and-evidence.md), & [IA-RAG Temporal QA (arXiv:2606.06044)](https://arxiv.org/html/2606.06044v1)
- **Abstention, Self-Correction, Verification & Multi-Turn Context:**
  - [Huang et al. (ICLR 2024) — Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/pdf?id=IkmD3fKBPQ)
  - [Meta CRAG Benchmark Analysis (arXiv:2406.04744v2)](https://akillness.github.io/posts/crag-new-rag-benchmark-dataset/) & [EnterpriseRAG (arXiv:2608.11584)](https://arxiv.org/html/2608.11584)
  - [TruthRL — Ternary Abstention Reward (OpenReview)](https://openreview.net/forum?id=KUmGzZo9am)
  - [Operating-Point Audit of Hallucination Detectors (MiniCheck / LettuceDetect)](https://towardsdatascience.com/ten-is-not-a-hundred/) & [TU Wien (2026) — Latency-Tiered Hallucination Detection](https://repositum.tuwien.at/bitstream/20.500.12708/229224/1/Rathmayr%20Michael%20-%202026%20-%20Latency-Tiered%20Hallucination%20Detection%20Optimizing...pdf)
  - [Liu et al. — Lost in the Middle](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf), [Context Rot Notes](https://olegchursin.com/notes/context-rot), & [TopiOCQA Over-Carry Analysis (OpenReview)](https://openreview.net/pdf?id=UbJKvxKAmZ)
- **Indirect Prompt Injection & Archive Security:**
  - [Spotlighting Analysis (Hines et al., 2024)](https://www.omarfaroque.com/posts/xpia_spotlighting.html)
  - [Large-Scale Public Red-Teaming Review of IPI Defenses (OpenReview)](https://openreview.net/pdf/8cd8c535ef133c362804a271b03ca3066df6be75.pdf)
  - [INFUSE Unified Surface IPI Benchmark](https://www.researchgate.net/publication/415932830_INFUSE_A_Unified_Surface_Benchmark_for_Indirect_Prompt_Injection_in_Agentic_Systems) & [Odile Representation Defense (COLM 2026)](https://proceedings.colm.cc/paper_files/paper/2026/file/c33cd281f8cd784626a57de340e43fe4-Paper-Conference.pdf)
  - [PDF Parser Divergence & V/AP Injection](https://pqpdf.com/pdf-ai-ingestion.php) & [Archive Decompression Bomb / Zip Slip Advisories (IBM)](https://www.ibm.com/support/pages/node/7285931)