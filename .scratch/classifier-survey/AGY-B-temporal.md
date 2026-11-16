# Candidate Classifiers, Routers, and Detectors

---

### 1. SituatedQA Context-Dependence Classifier
* **Authors / Year**: Michael J.Q. Zhang and Eunsol Choi (2021) [READ](file:///home/bertie/.gemini/antigravity-cli/brain/acdbb958-3b09-4d24-a010-aa12de98ff48/.system_generated/steps/20/content.md).
* **URL**: [arXiv:2109.06157](https://arxiv.org/abs/2109.06157) [READ]; [GitHub: mikejqzhang/SituatedQA](https://github.com/mikejqzhang/SituatedQA) [READ].
* **What it predicts (exact labels)**: Binary classification: `true` (context-dependent) vs `false` (static/context-independent) [READ], evaluated separately for temporal (`TEMP`) and geographic (`GEO`) dependencies.
* **Model & Parameter Count**: Fine-tuned `BERT-base` (110M parameters) and `BERT-large` (340M parameters) using pooled `[CLS]` token representations [SNIPPET].
* **Trained weights released & Licence**: **NOT FOUND** (no pre-trained binary classifier weights released on Hugging Face or GitHub; only dataset and training code released) [READ]. Base BERT models are Apache-2.0. Repository code is Apache-2.0 [SNIPPET]; dataset is CC BY-SA 4.0 [SNIPPET].
* **Training data & size**: 4,438 training questions for `TEMP` (plus 2,572 dev, 1,962 test; 8,972 total) and 1,149 training questions for `GEO` (879 dev, 367 test; 2,395 total) extracted from Natural Questions, WebQuestions, TyDi QA, and MS MARCO [SNIPPET]. Training set is released [READ].
* **Reported Accuracy / F1**: On the `TEMP` test set (1,962 questions), `BERT-large` achieved **88.5% Accuracy / F1** (93.2% dev) [SNIPPET]. Breakdown on test subsets: Static: 93.8%, Sampled: 91.2%, Start: 85.9% [SNIPPET].
* **Latency**: **NOT FOUND** (unreported in paper; standard PyTorch `BERT-base` CPU inference takes ~20–35 ms, `BERT-large` takes ~70–120 ms).
* **Handles time-sensitivity/freshness**: Yes; explicitly predicts whether an answer changes over time (`TEMP`) [READ].
* **Verdict against constraints**: **Usable after fine-tuning**. The authors did not release trained checkpoint weights, but the 4,438-example `TEMP` training dataset is released. Fine-tuning an efficient encoder (e.g., `all-MiniLM-L6-v2` or `deberta-v3-small`) fits CPU latency (<25 ms) and footprint (<150 MB).

---

### 2. Meta CRAG Dynamism Router (KDD Cup 2024 Solution – Team APEX)
* **Authors / Year**: Jie Ouyang, Yucong Luo, Mingyue Cheng, et al. / Team APEX (2024) [READ](file:///home/bertie/.gemini/antigravity-cli/brain/acdbb958-3b09-4d24-a010-aa12de98ff48/.system_generated/steps/130/content.md).
* **URL**: [arXiv:2409.15337](https://arxiv.org/abs/2409.15337) [READ]; [GitHub: USTCAGI/CRAG-in-KDD-Cup2024](https://github.com/USTCAGI/CRAG-in-KDD-Cup2024) [READ].
* **What it predicts (exact labels)**: 4 temporal dynamism classes: `static`, `slow-changing`, `fast-changing`, and `real-time` [READ]. (Accompanied by a 5-class Domain Router: `finance`, `sports`, `music`, `movie`, `open domain` [READ]).
* **Model & Parameter Count**: Fine-tuned sequence classifier based on `BAAI/bge-m3` (567M parameters, XLM-RoBERTa architecture) [READ].
* **Trained weights released & Licence**: **NOT FOUND** as standalone Hugging Face checkpoints; weights directory `models/router/bge-m3/dynamic/` in the repository holds configuration files, but trained checkpoint tensors are omitted/unhosted [READ]. Base `BAAI/bge-m3` is MIT [READ]. Repository is MIT [READ].
* **Training data & size**: Trained on the Meta CRAG benchmark training split (4,409 questions annotated with ground-truth dynamism and domain labels, available via AIcrowd and `facebookresearch/CRAG`) [READ].
* **Reported Accuracy / F1**: Standalone router classification accuracy was **NOT FOUND** in the technical report; the overall adaptive RAG pipeline finished **2nd place** in Tasks 2 & 3 of Meta KDD Cup 2024 [READ]. (Note: 1st place team `db3` from Peking University did not use a dynamism question classifier [SNIPPET]).
* **Latency**: **NOT FOUND** in report. On laptop CPU, `bge-m3` (567M params, FP16 ~1.14 GB, FP32 ~2.2 GB) takes ~90–160 ms per query, exceeding our 50 ms budget.
* **Handles time-sensitivity/freshness**: Yes; specifically targets rate of temporal change [READ].
* **Verdict against constraints**: **Usable after fine-tuning** (only with a smaller backbone). The 567M BGE-M3 router violates both the 500 MB footprint and 50 ms CPU budget. However, the 4,409 CRAG questions can be used to fine-tune a small encoder (<50M params).

---

### 3. NTCIR Temporalia Temporal Query Intent Classifier (TUTA1)
* **Authors / Year**: NTCIR-11 Temporalia organizers (Hideo Joho et al., 2014); best system by Team TUTA1 (Univ. of Tokushima & Tongji Univ., 2014) [SNIPPET].
* **URL**: [NTCIR-11 Temporalia Overview](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings11/pdf/NTCIR/OVERVIEW/01-NTCIR11-OV-TEMPORALIA-JohoH.pdf) [SNIPPET]; open implementation: [GitHub: filannim/Temporalia-TQIC](https://github.com/filannim/Temporalia-TQIC) [READ].
* **What it predicts (exact labels)**: 4 temporal intent classes: `Atemporal`, `Past`, `Recency`, `Future` [SNIPPET].
* **Model & Parameter Count**: Linear supervised/semi-supervised classifiers (SVM / Logistic Regression, <1M parameters) utilizing rule-based time-gap calculations, verb tenses, lemmas, and named entities [SNIPPET].
* **Trained weights released & Licence**: **NOT FOUND** (no pre-trained model checkpoint released) [SNIPPET]. NTCIR data requires signed academic agreement [SNIPPET]; `filannim/Temporalia-TQIC` implementation is Apache-2.0 [READ].
* **Training data & size**: NTCIR-11 TQIC Dry Run: 100 queries [SNIPPET]; Formal Test Run: 300 queries [SNIPPET]. NTCIR-12 TID Dry Run: 93 English queries [SNIPPET]; Formal Run: 300 queries [SNIPPET].
* **Reported Accuracy / F1**: Team TUTA1 best run (`TUTA1-TQIC-RUN-1`) achieved **74.0% accuracy** on the 300 test queries (Past: 85.3%, Future: 85.3%, Atemporal: 77.3%, Recency: 48.0%) [SNIPPET]. The `filannim/Temporalia-TQIC` pipeline achieved 66.3% accuracy [READ].
* **Latency**: **NOT FOUND** (rule/linear pipeline takes <5 ms on CPU).
* **Handles time-sensitivity/freshness**: Yes, separates `Recency` and `Future` from `Atemporal` and `Past` [SNIPPET].
* **Verdict against constraints**: **Only the idea**. The 4-class taxonomy is influential, but the training dataset is tiny (100–300 search keyword strings), non-commercial/restricted, and modern transformer encoders outperform hand-engineered verb/time-gap regexes.

---

### 4. Query Temporal Profilers & Log-Based Detectors
* **Authors / Year**: Rosie Jones & Fernando Diaz (2007) [SNIPPET]; Anagha Kulkarni et al. (WSDM 2011) [SNIPPET]; Wisam Dakka et al. (CIKM 2008 / 2012) [SNIPPET].
* **URL**: [Jones & Diaz, ACM TOIS 2007](https://dl.acm.org/doi/10.1145/1247480.1247484) [SNIPPET]; [Kulkarni et al., WSDM 2011](https://dl.acm.org/doi/10.1145/1935826.1935847) [SNIPPET].
* **What it predicts (exact labels)**: Query temporal profile: `Atemporal`, `Temporally Unambiguous`, `Temporally Ambiguous` (Jones & Diaz) [SNIPPET], or binary time-sensitivity / temporal drift (Kulkarni et al.) [SNIPPET].
* **Model & Parameter Count**: Statistical time-series models (Fourier transforms, autocorrelation, burst detection) and gradient-boosted decision trees over search query logs and retrieved document timestamp distributions [SNIPPET].
* **Trained weights released & Licence**: **NOT FOUND** / Not applicable (proprietary search-engine log features from Yahoo/Bing/MSN) [SNIPPET].
* **Training data & size**: Millions of query-log records and TREC news collections (AP, WSJ) [SNIPPET]. Unreleased proprietary logs.
* **Reported Accuracy / F1**: Evaluated on mean average precision (MAP) gains and query drift correlation, not standalone question F1 [SNIPPET].
* **Latency**: **NOT FOUND**; requires querying an index or log database first.
* **Handles time-sensitivity/freshness**: Yes [SNIPPET].
* **Verdict against constraints**: **Only the idea**. Requires pre-retrieval access to search engine logs or post-retrieval document timestamps, making it unusable as a 50 ms pre-retrieval standalone question gate.

---

### 5. FreshCache Freshness Risk Gate
* **Authors / Year**: Muhammad Mansoor, Tahir Ahmad, and Yeo-Chan Yoon (2024/2026) [SNIPPET].
* **URL**: [arXiv:2607.04281](https://arxiv.org/abs/2607.04281) [SNIPPET].
* **What it predicts (exact labels)**: 5 freshness classes: `timeless`, `slow`, `medium`, `fast`, `real-time` [SNIPPET], mapping into continuous staleness probabilities via an error budget.
* **Model & Parameter Count**: A lightweight text classifier paired with a 25-dimensional Multilayer Perceptron (MLP) risk model (<500K parameters) [SNIPPET].
* **Trained weights released & Licence**: **NOT FOUND** on Hugging Face (preprint stage) [SNIPPET].
* **Training data & size**: Evaluated on *FreshCache-Bench* (thousands of queries evaluated across web snapshots at 1h, 12h, 24h, and 7d horizons) [SNIPPET].
* **Reported Accuracy / F1**: Achieved up to 97% search API cost savings while strictly bounding stale error rates to ~0.1% [SNIPPET].
* **Latency**: <5 ms for feature evaluation and MLP scoring [SNIPPET].
* **Handles time-sensitivity/freshness**: Yes [SNIPPET].
* **Verdict against constraints**: **Only the idea**. Weights and dataset are not currently published, but using an MLP or small encoder over freshness tiers directly matches our target gate architecture.

---

### 6. Small Pretrained Encoders for Fine-Tuned Gating
* **Models**:
  1. [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) [READ]: 22.7M parameters, ~90 MB footprint (FP32), Apache-2.0. CPU latency: ~8–15 ms per question.
  2. [`microsoft/deberta-v3-small`](https://huggingface.co/microsoft/deberta-v3-small) [READ]: 44M parameters, ~175 MB footprint, MIT. CPU latency: ~20–30 ms per question.
  3. [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) [READ]: 66.4M parameters, ~260 MB footprint, Apache-2.0. CPU latency: ~18–28 ms per question.
* **Trained weights for temporal gating**: **NOT FOUND** (no existing public checkpoint fine-tuned specifically on temporal question gating).
* **Verdict against constraints**: **Usable after fine-tuning**. Fully compliant with latency (<30 ms on CPU), footprint (<200 MB), permissive licensing (MIT/Apache-2.0), and zero LLM calls.

---

# Datasets We Could Train and Evaluate a Gate On

| Dataset Name | Size (Questions) | Labels / Classes | Licence | URL / Access |
| :--- | :--- | :--- | :--- | :--- |
| **SituatedQA (TEMP split)** | 8,972 total (4,438 train, 2,572 dev, 1,962 test) [SNIPPET] | Binary: `true` (temporal context-dependent / answer changes over time) vs `false` (static) [READ] | CC BY-SA 4.0 [SNIPPET] | [Hugging Face: siyue/SituatedQA](https://huggingface.co/datasets/siyue/SituatedQA) [READ]; [GitHub](https://github.com/mikejqzhang/SituatedQA) [READ] |
| **Meta CRAG (Comprehensive RAG)** | 4,409 QA pairs [READ] | 4 Dynamism classes: `static`, `slow-changing`, `fast-changing`, `real-time` [READ]; plus 5 Domain classes [READ] | CC BY-NC 4.0 [READ] | [GitHub: facebookresearch/CRAG](https://github.com/facebookresearch/CRAG) [READ] |
| **FreshQA** | 600 questions [READ] | 4 Classes: `fast-changing`, `slow-changing`, `never-changing`, `false-premise` [READ] | Apache-2.0 [READ] | [GitHub: freshllms/freshqa](https://github.com/freshllms/freshqa) [READ] |
| **RealTimeQA** | Ongoing weekly releases (~30 questions/week since 2022; thousands total) [SNIPPET] | Positive news questions (all time-sensitive current events; no static negative class) [READ] | MIT (code/public repository) [READ] | [GitHub: realtimeqa/realtimeqa_public](https://github.com/realtimeqa/realtimeqa_public) [READ] |
| **TimeQA (Time-Sensitive QA)** | ~10k+ questions across Easy and Hard splits [SNIPPET] | Question-answer-context spans with explicit temporal constraints (no static/changing binary gate labels) [READ] | BSD 3-Clause [READ] | [GitHub: wenhuchen/Time-Sensitive-QA](https://github.com/wenhuchen/Time-Sensitive-QA) [READ] |
| **SealQA** | 3 subsets: `Seal-0`, `Seal-Hard`, `LongSeal` [SNIPPET] | Fact-seeking questions designed for unhelpful/conflicting search results (evaluation benchmark) [SNIPPET] | Apache-2.0 [SNIPPET] | [Hugging Face: vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa) [SNIPPET] |
| **FreshStack** | ~500–1,000 developer questions across 5 technical topics [SNIPPET] | Topic/versioned software queries from Stack Overflow (code-heavy, not conversational voice queries) [SNIPPET] | CC-BY-SA 4.0 (data) / Apache-2.0 (code) [SNIPPET] | [GitHub: fresh-stack/freshstack](https://github.com/fresh-stack/freshstack) [SNIPPET] |

---

# Ranking

### 1. Fine-Tuned Sequence Classifier on `sentence-transformers/all-MiniLM-L6-v2`
* **Plan (3 lines)**:
  1. Map existing labels into binary target classes: Label `1` (needs web search) for SituatedQA `true`, CRAG (`slow-changing`, `fast-changing`, `real-time`), and FreshQA (`fast-changing`, `slow-changing`); Label `0` (internal answer) for SituatedQA `false`, CRAG `static`, and FreshQA `never-changing`.
  2. Fine-tune a sequence classification head on `all-MiniLM-L6-v2` (22.7M parameters) using the combined ~9,400 training examples with class-weighted cross-entropy loss.
  3. Export to ONNX Runtime with INT8 quantization, giving ~10 ms latency on a single laptop CPU thread, ~25 MB disk footprint, and Apache-2.0 licence.
* **What would make us change the order**: If classification accuracy on subtle slow-changing role/pricing queries is insufficient, upgrading to `microsoft/deberta-v3-small` (44M params, ~25 ms CPU) provides stronger disentangled attention representations at modest latency cost.

---

### 2. Few-Shot Sentence-Transformer Gate using SetFit
* **Plan (3 lines)**:
  1. Select 50 balanced, high-diversity exemplars per class from FreshQA and CRAG across `real-time`, `fast-changing`, `slow-changing`, and `static` queries.
  2. Train a SetFit model (contrastive sentence fine-tuning + logistic regression head) on `all-MiniLM-L6-v2` in under 3 minutes on CPU without full backpropagation tuning.
  3. Deploy the resulting model as an embedded scikit-learn/sentence-transformers pipeline executing in ~12 ms on laptop CPU.
* **What would make us change the order**: If labeled training data must be updated frequently as new domains arise without running a full training loop, or if we have fewer than 200 labeled examples, SetFit moves to #1.

---

### 3. Two-Tier Cascaded Gate (Fast Regex + DeBERTa-v3-small Fallback)
* **Plan (3 lines)**:
  1. Keep an ultra-high-precision regex fast-path (<1 ms) for unmistakable temporal triggers (e.g., "today", "current price of", "who is the current", "latest version").
  2. Queries that bypass the fast-path regex pass to a fine-tuned `microsoft/deberta-v3-small` classifier (44M parameters) running in ~22 ms on CPU.
  3. Tune the classifier decision threshold $\tau$ (e.g., $\tau = 0.35$) to prioritize recall on time-sensitive facts while bounding false alarms on never-changing queries below 10%.
* **What would make us change the order**: If turn latency budget is squeezed to <15 ms total for the entire routing step, the cascaded hybrid becomes #1 because obvious queries bypass the neural model completely.
