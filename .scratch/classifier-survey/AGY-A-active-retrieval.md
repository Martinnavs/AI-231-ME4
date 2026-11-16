# When-to-Retrieve Classifiers & Routing Survey for Spoken Voice Assistants

## 1. Candidate Classifiers, Routers, and Detectors

---

### **UAR (Unified Active Retrieval)**
* **Authors / Year:** Qinyuan Cheng et al., 2024 (Findings of EMNLP 2024) [SNIPPET].
* **URLs:** Paper: [arXiv:2406.12534](https://arxiv.org/abs/2406.12534) [SNIPPET]; Code: [xiami2019/UAR](https://github.com/xiami2019/UAR) [SNIPPET].
* **Predictions (Exact Labels):** Four orthogonal binary classifiers: Intent-aware, Knowledge-aware, Time-sensitive/Time-aware, and Self-aware. Each predicts `{0: no retrieval, 1: retrieval needed}` [SNIPPET].
* **Model & Parameters:** A single-layer Multi-Layer Perceptron (MLP, `hidden_dim -> 2`, ~8K params) trained on the last-token hidden state of a frozen base LLM (e.g., `meta-llama/Llama-2-7b-chat-hf`, 7B params) [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND**. Pre-trained model checkpoints/weights are not hosted on Hugging Face or GitHub [SNIPPET]. Repo licence: **NOT FOUND** (no `LICENSE` file in repository root; unclear / default all-rights-reserved) [SNIPPET].
* **Training Data & Size:** AR-Bench dataset (~8,000 samples per criterion, 1:1 balanced ratio). The time-aware subset (`time_aware_cls_ce`) pairs time-evolving questions (sourced from temporal QA like TAQA) against static questions from TriviaQA [SNIPPET]. Released in `training_data.zip` inside the GitHub repo [SNIPPET].
* **Reported Accuracy / F1:** 86.69% accuracy on the AR-Bench Time-aware test set (Table 2 of paper) using LLaMA-2-7B [SNIPPET].
* **Latency:** MLP head alone is <1 ms, but requires a full forward pass through LLaMA-2-7B (>1,000 ms on a laptop CPU) [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** Yes, explicitly trained on a time-sensitivity axis [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Needs LLM hidden states** (last-token hidden vector) [SNIPPET]. Incompatible with hosted Gemini 3.1 flash-lite.
* **Verdict:** **Only the idea** (requires base LLM internal hidden states; violates latency, footprint, and API constraints).

---

### **SKR (Self-Knowledge Guided Retrieval)**
* **Authors / Year:** Yile Wang, Peng Li, Maosong Sun, Yang Liu, 2023 (Findings of EMNLP 2023) [SNIPPET].
* **URLs:** Paper: [arXiv:2310.05002](https://arxiv.org/abs/2310.05002) [SNIPPET]; Code: [THUNLP-MT/SKR](https://github.com/THUNLP-MT/SKR) [SNIPPET].
* **Predictions (Exact Labels):** Binary: `{known (no retrieval), unknown (retrieve)}` [SNIPPET].
* **Model & Parameters:** Prompting, $k$-NN text embeddings, or a trained BERT/RoBERTa classifier (`SKR-cls`, ~110M–355M params) [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND**. No pre-trained weights published on Hugging Face [SNIPPET]. Repo licence: **NOT FOUND** (no `LICENSE` file in repo root; unclear) [SNIPPET].
* **Training Data & Size:** Question correctness history from PopQA, StrategyQA, and HotpotQA (~tens of thousands of static samples) [SNIPPET].
* **Reported Accuracy / F1:** Evaluated by Cheng et al. (2024) on AR-Bench Time-aware: 76.61% accuracy [SNIPPET]. On PopQA, improves QA accuracy by skipping retrieval on known questions [SNIPPET].
* **Latency:** BERT-base classification runs in ~25–35 ms on laptop CPU [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No; only checks whether static facts are known to the model's pre-training memory [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Works from question text alone** (encodes text via RoBERTa/BERT) [SNIPPET].
* **Verdict:** **Usable after fine-tuning** (conceptually fits CPU budget from text alone, but lacks released weights and was trained on static facts rather than freshness).

---

### **RA-ISF (Retrieval Augmented Iterative Self-Feedback)**
* **Authors / Year:** Yanming Liu et al., 2024 (Findings of ACL 2024) [SNIPPET].
* **URLs:** Paper: [arXiv:2403.06840](https://arxiv.org/abs/2403.06840) [SNIPPET]; Code: [OceannTwT/ra-isf](https://github.com/OceannTwT/ra-isf) [SNIPPET].
* **Predictions (Exact Labels):** Multi-stage: Self-Knowledge (`know` vs. `unknow`), Passage Relevance (`relevant` vs. `irrelevant`), and query sub-decomposition [SNIPPET].
* **Model & Parameters:** Fine-tuned 7B LLM (LLaMA-2-7B-chat) or prompted GPT-3.5 [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** on Hugging Face [SNIPPET]. Repo data/artifacts released under **CC BY-NC-SA 4.0 (Non-Commercial)**; code root lacks an explicit `LICENSE` file [SNIPPET].
* **Training Data & Size:** Multi-hop datasets (HotpotQA, 2WikiMultiHopQA) with synthesized critique labels [SNIPPET].
* **Reported Accuracy / F1:** Evaluated on downstream generation EM/F1 across multi-hop reasoning; standalone freshness gating accuracy: **NOT FOUND** [SNIPPET].
* **Latency:** Iterative multi-turn prompting requires 3–10 s per query [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No (focuses on multi-hop static logic) [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Calls an LLM** (multiple iterative generation calls) [SNIPPET].
* **Verdict:** **Only the idea** (heavy LLM prompting loop, violates latency, no-LLM, and commercial constraints).

---

### **Rowen**
* **Authors / Year:** Hanxing Ding, Liang Pang, Zihao Wei, Xueqi Cheng, 2024 (arXiv:2402.10612 / SIGIR-AP) [SNIPPET].
* **URLs:** Paper: [arXiv:2402.10612](https://arxiv.org/abs/2402.10612) [SNIPPET]; Code: [dhx20150812/Rowen](https://github.com/dhx20150812/Rowen) [SNIPPET].
* **Predictions (Exact Labels):** Continuous multilingual semantic consistency score thresholded to decide `{retrieve, skip-retrieval}` [SNIPPET].
* **Model & Parameters:** 7B–13B LLM (LLaMA-2) [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** (training-free framework) [SNIPPET]. Repo licence: **NOT FOUND** [SNIPPET].
* **Training Data & Size:** Training-free [SNIPPET].
* **Reported Accuracy / F1:** Evaluated on PopQA and TriviaQA generation; no standalone time-sensitivity metric [SNIPPET].
* **Latency:** Requires translating the input question and executing multiple LLM generation passes (>3 s) [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No (targets parametric factual hallucination) [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Calls an LLM** multiple times [SNIPPET].
* **Verdict:** **Only the idea** (requires multiple full LLM generations, violates 50 ms budget).

---

### **Probing-RAG**
* **Authors / Year:** Ingeol Baek, Hwan Chang, Byeongjeong Kim, Jimin Lee, Hwanhee Lee, 2024 (Findings of NAACL 2025) [SNIPPET].
* **URLs:** Paper: [arXiv:2410.13339](https://arxiv.org/abs/2410.13339) [SNIPPET]; Code: [baekingeol/Probing-RAG](https://github.com/baekingeol/Probing-RAG) [SNIPPET].
* **Predictions (Exact Labels):** Retrieval necessity `{retrieve, no-retrieve}` and retrieval hop count [SNIPPET].
* **Model & Parameters:** Linear / MLP probe (~10K params) inspecting intermediate hidden states of a 7B LLM [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** on Hugging Face [SNIPPET]. Repo licence: **NOT FOUND** [SNIPPET].
* **Training Data & Size:** Hidden-state activations on NQ, PopQA, and TriviaQA [SNIPPET].
* **Reported Accuracy / F1:** Evaluated on selective QA accuracy on NQ/TriviaQA; freshness metric: **NOT FOUND** [SNIPPET].
* **Latency:** Probe is <1 ms, but requires intermediate LLM layer execution (>1 s on CPU) [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Needs LLM hidden states** (intermediate layer representations) [SNIPPET]. Incompatible with hosted Gemini.
* **Verdict:** **Only the idea** (requires internal layer access to the LLM).

---

### **SeaKR (Self-Aware Knowledge Retrieval)**
* **Authors / Year:** Zijun Yao et al., 2024 (arXiv:2406.19215) [SNIPPET].
* **URLs:** Paper: [arXiv:2406.19215](https://arxiv.org/abs/2406.19215) [SNIPPET]; Code: [AdaRAGUE/SeaKR](https://github.com/AdaRAGUE/SeaKR) [SNIPPET].
* **Predictions (Exact Labels):** Continuous self-aware uncertainty thresholded into `{retrieve, skip}` [SNIPPET].
* **Model & Parameters:** Base LLM (LLaMA-2-7B / Mistral-7B) [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** (training-free) [SNIPPET]. Repo licence: **NOT FOUND** [SNIPPET].
* **Training Data & Size:** Training-free; evaluated on PopQA, TriviaQA, StrategyQA [SNIPPET].
* **Reported Accuracy / F1:** Outperforms static retrieval on PopQA/TriviaQA; freshness metric: **NOT FOUND** [SNIPPET].
* **Latency:** Requires monitoring LLM generation dynamics and internal states (>1 s on CPU) [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Needs LLM hidden states** and output dynamics [SNIPPET].
* **Verdict:** **Only the idea** (incompatible with hosted Gemini API).

---

### **DRAGIN (with RIND)**
* **Authors / Year:** Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, Yiqun Liu, 2024 (ACL 2024) [SNIPPET].
* **URLs:** Paper: [arXiv:2403.10081](https://arxiv.org/abs/2403.10081) [SNIPPET]; Code: [WeihangSu/DRAGIN](https://github.com/WeihangSu/DRAGIN) [SNIPPET].
* **Predictions (Exact Labels):** Token-level Real-time Information Need Detection (RIND): `{trigger retrieval, continue generation}` [SNIPPET].
* **Model & Parameters:** Base LLM (LLaMA-2-7B/13B) [SNIPPET].
* **Trained Weights & Licence:** Released code under **Apache-2.0** [SNIPPET].
* **Training Data & Size:** Training-free [SNIPPET].
* **Reported Accuracy / F1:** Evaluated on multi-hop QA (2WikiMultiHopQA, HotpotQA, IIRC) [SNIPPET].
* **Latency:** Evaluated token-by-token during generation [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Needs LLM logits and self-attention states** [SNIPPET]. Incompatible with hosted Gemini.
* **Verdict:** **Only the idea** (requires token entropy and attention maps from the generating model).

---

### **Self-DC (Self Divide-and-Conquer)**
* **Authors / Year:** Hongru Wang et al., 2024 (arXiv:2402.13514 / NAACL 2025) [SNIPPET].
* **URLs:** Paper: [arXiv:2402.13514](https://arxiv.org/abs/2402.13514) [SNIPPET].
* **Predictions (Exact Labels):** Sub-question routing: `{reason (internal parametric), act (external retrieval)}` [SNIPPET].
* **Model & Parameters:** Prompted LLMs (GPT-3.5, LLaMA-2-7B) [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** [SNIPPET].
* **Training Data & Size:** Evaluated on CuQA (Compositional unknown Question Answering) [SNIPPET].
* **Reported Accuracy / F1:** Evaluated on CuQA accuracy; freshness metric: **NOT FOUND** [SNIPPET].
* **Latency:** Multi-step prompt decomposition requires >3 s [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Calls an LLM** (multiple prompt steps) [SNIPPET].
* **Verdict:** **Only the idea** (violates latency and no-LLM budget).

---

### **EG-E5 / EverGreen (`s-nlp/E5-EverGreen-Multilingual-Small`)**
* **Authors / Year:** Sergey Pletenev et al., 2025 (arXiv:2505.21115 / EMNLP) [SNIPPET].
* **URLs:** Paper: [arXiv:2505.21115](https://arxiv.org/abs/2505.21115) [SNIPPET]; Hugging Face: [s-nlp/E5-EverGreen-Multilingual-Small](https://huggingface.co/s-nlp/E5-EverGreen-Multilingual-Small) [SNIPPET]; Code: [s-nlp/EverGreen-classification](https://github.com/s-nlp/EverGreen-classification) [SNIPPET].
* **Predictions (Exact Labels):** Binary: `{evergreen, mutable}` (`evergreen` = fact never changes; `mutable` = fact changes over time) [SNIPPET].
* **Model & Parameters:** Fine-tuned `intfloat/multilingual-e5-small`, **118M parameters** (~470 MB FP32, ~118 MB INT8) [SNIPPET].
* **Trained Weights & Licence:** **RELEASED** on Hugging Face: `s-nlp/E5-EverGreen-Multilingual-Small` (and `Large`). Licence: **MIT** [SNIPPET].
* **Training Data & Size:** EverGreenQA (`s-nlp/EverGreen-Multilingual`), 4,757 human-curated questions (3,487 train, 1,270 test) across 7 languages (EN, ZH, FR, DE, RU, HE, AR), balanced with 1,449 synthetic samples. Released under MIT [SNIPPET].
* **Reported Accuracy / F1:** **0.88 accuracy, 0.89 overall F1** (0.92 English F1) on the blind EverGreenQA test set [SNIPPET].
* **Latency:** ~15–25 ms on a laptop CPU in FP32; sub-15 ms quantized [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** **Yes, exactly** (trained specifically to separate mutable changing facts from static evergreen facts) [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Works from question text alone** [SNIPPET].
* **Verdict:** **Usable as is** (fully matches latency, footprint, text-only input, licence, and freshness requirements).

---

### **Ammu-1.0-FactCheck-Need (`kalyan-ks/Ammu-1.0-FactCheck-Need-small`)**
* **Authors / Year:** Kalyan KS, 2025 [SNIPPET].
* **URLs:** Hugging Face: [kalyan-ks/Ammu-1.0-FactCheck-Need-small](https://huggingface.co/kalyan-ks/Ammu-1.0-FactCheck-Need-small) [SNIPPET]; Base: [kalyan-ks/Ammu-1.0-FactCheck-Need-base](https://huggingface.co/kalyan-ks/Ammu-1.0-FactCheck-Need-base) [SNIPPET].
* **Predictions (Exact Labels):** Binary: `{FACT_CHECK_NEEDED, NO_FACT_CHECK_NEEDED}` [SNIPPET].
* **Model & Parameters:** ModernBERT (Ettin Encoder): Small is **68M parameters** (~130 MB); Base is 150M parameters (~285 MB INT8, ~571 MB FP32); Tiny is 32M parameters (~60 MB) [SNIPPET].
* **Trained Weights & Licence:** **RELEASED** on Hugging Face: `kalyan-ks/Ammu-1.0-FactCheck-Need-small` and `kalyan-ks/Ammu-1.0-FactCheck-Need-base`. Licence: **MIT** [SNIPPET].
* **Training Data & Size:** Synthetic dataset across 120 topics and 19 formats; evaluated on 4,214 blind test set (`kalyan-ks/fact-check-need-classification`) [SNIPPET].
* **Reported Accuracy / F1:** 96.56% Macro-F1 on test set (Base); ~95% Macro-F1 (Small) [SNIPPET].
* **Latency:** CPU P50 latency: Base ONNX-INT8 is 35.73 ms; Small is ~18 ms; Tiny is <10 ms [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** Partially (detects if external verification is required, but does not split static facts from changing facts) [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Works from question text alone** [SNIPPET].
* **Verdict:** **Usable as is** for general fact-check routing, or **usable after fine-tuning** on FreshQA/SituatedQA for time-sensitivity.

---

### **RAGate**
* **Authors / Year:** Xi Wang, Procheta Sen, Ruizhe Li, Emine Yilmaz, 2024 (Findings of NAACL 2025 / arXiv:2407.21712) [SNIPPET].
* **URLs:** Paper: [arXiv:2407.21712](https://arxiv.org/abs/2407.21712) [SNIPPET]; Code: [wangxieric/RAGate](https://github.com/wangxieric/RAGate) [SNIPPET].
* **Predictions (Exact Labels):** Binary conversational turn label: `{0: no retrieval, 1: retrieval needed}` [SNIPPET].
* **Model & Parameters:** Multi-Head Attention (MHA) neural encoder on top of BERT/RoBERTa (~110M params) or PEFT LLaMA-2-7B [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** on Hugging Face [SNIPPET]. Code licensed under **MIT** [SNIPPET].
* **Training Data & Size:** Dialogue turns adapted from conversational benchmarks (e.g. Wizard of Wikipedia) [SNIPPET].
* **Reported Accuracy / F1:** PEFT achieves F1 ~0.4082 on conversational selective augmentation [SNIPPET].
* **Latency:** BERT-MHA is ~25–35 ms on CPU; PEFT 7B is >1 s [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No (distinguishes knowledge-grounded dialogue from chitchat) [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Works from dialogue text alone** [SNIPPET].
* **Verdict:** **Usable after fine-tuning** (code is MIT and encoder is small, but lacks released weights for freshness).

---

### **KBM (Knowledge Boundary Model)**
* **Authors / Year:** Zhen Zhang et al., 2024 (Findings of EMNLP 2024 / arXiv:2411.06207) [SNIPPET].
* **URLs:** Paper: [arXiv:2411.06207](https://arxiv.org/abs/2411.06207) [SNIPPET].
* **Predictions (Exact Labels):** Boundary classification: `{inside boundary (no retrieval), outside boundary (retrieval needed)}` [SNIPPET].
* **Model & Parameters:** Analyzes base LLM generative confidence / internal states [SNIPPET].
* **Trained Weights & Licence:** **NOT FOUND** (no weights or repository published) [SNIPPET].
* **Training Data & Size:** QA boundary instances [SNIPPET].
* **Reported Accuracy / F1:** Reported downstream generation gains on static QA; freshness metric: **NOT FOUND** [SNIPPET].
* **Latency:** >1 s on CPU [SNIPPET].
* **Handles Freshness / Time-Sensitivity:** No [SNIPPET].
* **Hidden States vs. Question Text Alone:** **Needs LLM internal states** [SNIPPET].
* **Verdict:** **Only the idea** (no public weights; requires LLM internal states).

---

## 2. Datasets We Could Train and Evaluate a Gate On

1. **FreshQA**
   * **Size:** 600 questions [SNIPPET].
   * **Labels:** 4 classes: `never-changing`, `slow-changing`, `fast-changing`, `false-premise` (can be collapsed to binary: changing vs. never-changing) [SNIPPET].
   * **Licence:** Apache License 2.0 [SNIPPET].
   * **URL:** [freshllms/freshqa](https://github.com/freshllms/freshqa) [SNIPPET].

2. **SituatedQA**
   * **Size:** ~5,200 questions total (~3,500 temporal questions) [SNIPPET].
   * **Labels:** `static` (never-changing), `temporal` (TEMP, time-dependent), and `geographical` (GEO) [SNIPPET].
   * **Licence:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) [SNIPPET].
   * **URL:** [SituatedQA Project](https://situatedqa.github.io/) / [szrq/SituatedQA](https://github.com/szrq/SituatedQA) [SNIPPET].

3. **EverGreenQA (`s-nlp/EverGreen-Multilingual`)**
   * **Size:** 4,757 questions (3,487 train, 1,270 test) [SNIPPET].
   * **Labels:** Binary: `evergreen` (timeless/stable) vs. `mutable` (time-dependent/changing) across 7 languages [SNIPPET].
   * **Licence:** MIT License [SNIPPET].
   * **URL:** [s-nlp/EverGreen-Multilingual](https://huggingface.co/datasets/s-nlp/EverGreen-Multilingual) [SNIPPET].

4. **AR-Bench Time-Aware Subset (from UAR)**
   * **Size:** ~8,000 question samples (1:1 balanced ratio) [SNIPPET].
   * **Labels:** Binary: `time-sensitive` (`1`) vs. `static` (`0`) [SNIPPET].
   * **Licence:** Unclear / **NOT FOUND** (no `LICENSE` file in repo; default all-rights-reserved) [SNIPPET].
   * **URL:** [xiami2019/UAR](https://github.com/xiami2019/UAR) [SNIPPET].

5. **Fact-Check-Need-Classification (`kalyan-ks/fact-check-need-classification`)**
   * **Size:** 4,214 blind test samples + synthetic multi-topic train set [SNIPPET].
   * **Labels:** Binary: `FACT_CHECK_NEEDED` vs. `NO_FACT_CHECK_NEEDED` [SNIPPET].
   * **Licence:** MIT License [SNIPPET].
   * **URL:** [kalyan-ks/fact-check-need-classification](https://huggingface.co/datasets/kalyan-ks/fact-check-need-classification) [SNIPPET].

6. **PopQA**
   * **Size:** 14,267 questions annotated with Wikipedia monthly view counts [SNIPPET].
   * **Labels:** High-frequency/popular entities (parametrically known) vs. tail/obscure entities (retrieval required) [SNIPPET].
   * **Licence:** MIT License [SNIPPET].
   * **URL:** [AlexTMallen/adaptive-retrieval](https://github.com/AlexTMallen/adaptive-retrieval) [SNIPPET].

---

## 3. Ranking

### **Option 1: Deploy `s-nlp/E5-EverGreen-Multilingual-Small` (Zero-shot, then fine-tuned on FreshQA + SituatedQA)**
* **Concrete Plan:**
  1. Load `s-nlp/E5-EverGreen-Multilingual-Small` (118M params, MIT) in ONNX Runtime INT8 for ~15 ms CPU latency and ~118 MB RAM.
  2. Map `mutable` $\rightarrow$ trigger search and `evergreen` $\rightarrow$ skip search; benchmark immediately on FreshQA's 600 questions.
  3. Fine-tune for 3 epochs on the combined EverGreen (3.5k) + SituatedQA (3.5k) datasets to adjust decision margins for voice assistant queries.
* **What would change the order:** If zero-shot CPU latency on your laptop exceeds 35 ms, or if your voice assistant handles heavily informal speech where E5 tokenization degrades.

### **Option 2: Fine-tune `kalyan-ks/Ammu-1.0-FactCheck-Need-small` (ModernBERT-68M)**
* **Concrete Plan:**
  1. Take the 68M-parameter ModernBERT checkpoint (`kalyan-ks/Ammu-1.0-FactCheck-Need-small`, MIT), requiring only ~130 MB footprint and ~18 ms CPU latency.
  2. Add a second binary head or replace the classification head to predict `{CHANGING/OBSCURE, STATIC}`.
  3. Train on FreshQA + SituatedQA + PopQA tail entities so the model simultaneously catches both time-varying queries and obscure entities.
* **What would change the order:** If the voice assistant receives a high volume of non-factual conversational queries (chitchat, greetings, coding, math) where Ammu's pre-trained fact-check guardrail provides immediate upfront value over pure temporal detectors.

### **Option 3: Feature-Engineered Hybrid Classifier (TF-IDF + MiniLM Embeddings + LightGBM)**
* **Concrete Plan:**
  1. Extract fast surface indicators (temporal regex, POS-tag patterns, entity rarity cues) concatenated with sentence embeddings from `all-MiniLM-L6-v2` (22M params, <5 ms CPU).
  2. Train a LightGBM or Logistic Regression classifier on 8,000 merged questions (SituatedQA + EverGreen + FreshQA + PopQA tail).
  3. Calibrate the output probability threshold to cap false alarms on static questions at <10% while sustaining >85% recall on fast- and slow-changing questions.
* **What would change the order:** If RAM must stay under 50 MB, if latency must be strictly sub-5 ms, or if deployment requires a pure C++/NumPy pipeline without transformer runtime dependencies.
