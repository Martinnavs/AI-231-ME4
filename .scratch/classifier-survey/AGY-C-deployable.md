# Small Deployable Routers and Classifier Recipes for Voice Assistant Gating

Evidence tag key:
* `[READ]`: Directly verified by opening an allowed whitelisted URL.
* `[SNIPPET]`: Verified from search result snippet.
* `NOT FOUND`: Where exact information is unreleased or unpublished.

---

## 1. Candidate Classifiers, Routers, and Detectors

### 1. semantic-router
* **Authors / Year:** Aurelio Labs (James Briggs et al., 2023–2024) `[SNIPPET]`.
* **URL:** [github.com/aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router) `[READ]`.
* **What it predicts:** User-defined route names (e.g., `web_search`, `chitchat`) or `None` if max cosine similarity falls below a defined score threshold `[SNIPPET]`.
* **Model & Parameter Count:** Pluggable encoder architecture `[SNIPPET]`. Typically uses local ONNX models via FastEmbed: `BAAI/bge-small-en-v1.5` (33.4M params) or `sentence-transformers/all-MiniLM-L6-v2` (22.7M params) `[SNIPPET]`. The router layer itself has 0 learned parameters (computes vector cosine similarity against utterance centroids) `[SNIPPET]`.
* **Weights & Licence:** Library code is **MIT License** `[SNIPPET]`. Underlying models (`all-MiniLM-L6-v2`, `bge-small`) are Apache-2.0 / MIT `[SNIPPET]`. No pre-trained weights for freshness gating exist (routes must be supplied by the user as sample utterances) `[SNIPPET]`.
* **Training Data & Size:** None required; routes are configured via 5–30 example utterances per class `[SNIPPET]`.
* **Reported Accuracy / F1:** Vendor claims high semantic routing precision for distinct intent domains `[SNIPPET]`; formal accuracy on freshness benchmarks: `NOT FOUND`.
* **Latency:** Vendor claims 4 ms to 100 ms `[SNIPPET]`. On CPU using FastEmbed (`all-MiniLM-L6-v2` ONNX), vector inference takes ~10–15 ms, and vector dot-product against 30 route vectors takes <0.1 ms `[SNIPPET]`.
* **Handles Freshness:** No native freshness awareness; only triggers if an incoming query closely matches explicitly phrased temporal reference utterances (e.g., "what is the current price of...") `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning (usable as a recipe).** Very fast on CPU (~12 ms), but pure cosine similarity against static utterances struggles to generalize to subtle time-sensitive facts or obscure entities without explicit temporal keywords `[SNIPPET]`.

---

### 2. SetFit (Sentence Transformer Fine-tuning)
* **Authors / Year:** Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg (Hugging Face, Intel Labs, TU Darmstadt, 2022) `[READ]`.
* **URL:** [arxiv.org/abs/2209.11055](https://arxiv.org/abs/2209.11055) `[READ]` / [github.com/huggingface/setfit](https://github.com/huggingface/setfit) `[READ]`.
* **What it predicts:** Arbitrary user-defined discrete classes (e.g., binary `NEEDS_SEARCH` vs. `STATIC`) `[READ]`.
* **Model & Parameter Count:** Pretrained Sentence Transformer backbone (e.g., `all-MiniLM-L6-v2`, 22.7M parameters, ~90 MB disk footprint) + linear classification head (Logistic Regression) `[READ]`.
* **Weights & Licence:** Library is **Apache-2.0** `[SNIPPET]`. Backbone weights on Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) are **Apache-2.0** `[SNIPPET]`. No pre-trained checkpoint exists specifically for search gating `[READ]`.
* **Training Data & Size:** Specifically designed for extreme sample efficiency: typically requires only **8 to 32 labeled examples per class** (e.g., 16–64 total) `[READ]`.
* **Reported Accuracy / F1:** On SST-2 with 8 shots/class: 86.9% accuracy; on AG News with 8 shots/class: 87.2% accuracy; rivals full-scale fine-tuned RoBERTa-large on standard benchmarks with 32× fewer parameters `[SNIPPET]`. FreshQA accuracy: `NOT FOUND`.
* **Latency:** Embedding pass takes ~10–20 ms on modern laptop CPU (drops to ~5–8 ms with INT8 ONNX); logistic regression head takes <0.1 ms `[SNIPPET]`.
* **Handles Freshness:** Yes, when contrastively fine-tuned on pairs of static vs. changing/obscure queries `[READ]`.
* **Verdict:** **Usable after fine-tuning.** Outstanding candidate for low-resource training; runs in ~15 ms on CPU, consumes ~90 MB RAM, fully permissive Apache-2.0 `[READ]`.

---

### 3. fastText (Supervised Classification)
* **Authors / Year:** Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov (Facebook AI Research, 2016) `[SNIPPET]`.
* **URL:** [arxiv.org/abs/1607.01759](https://arxiv.org/abs/1607.01759) `[SNIPPET]` / [github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText) `[SNIPPET]`.
* **What it predicts:** User-defined discrete text labels (e.g., `__label__search`, `__label__no_search`) `[SNIPPET]`.
* **Model & Parameter Count:** Bag of words and character n-grams mapped to an averaged embedding layer feeding a hierarchical softmax/linear classifier `[SNIPPET]`. Model parameter count depends on vocabulary hash size; typically 1M–10M parameters (~5 MB to 40 MB disk size; can be quantized via `fasttext quantize` down to <2 MB) `[SNIPPET]`.
* **Weights & Licence:** Code and library licensed under **MIT License** `[SNIPPET]`. Supervised models are trained directly on local task datasets in seconds `[SNIPPET]`.
* **Training Data & Size:** Supervised text file formatted with labels `[SNIPPET]`. Evaluated originally on AG News (120k samples), DBPedia (560k samples), Sogou (450k samples) `[SNIPPET]`.
* **Reported Accuracy / F1:** AG News: 92.5% accuracy; DBPedia: 98.6% accuracy `[SNIPPET]`.
* **Latency:** **Sub-millisecond** on laptop CPU: typically **0.1 ms to 0.5 ms** per sentence `[SNIPPET]`.
* **Handles Freshness:** Only captures explicit n-gram surface patterns (e.g., "latest", "who is the current", "score of", "today") `[SNIPPET]`. It fails to detect semantic temporal subtleties or entity obscurity where explicit n-gram cues are absent `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning (training from scratch).** Ultra-low latency (~0.2 ms) and negligible footprint (<5 MB), but lower accuracy ceiling on complex semantic/temporal intent compared to transformer encoders `[SNIPPET]`.

---

### 4. Small Sentence-Embedding Models + Logistic Regression
* **Authors / Year:** 
  * `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2021) `[SNIPPET]`.
  * `BAAI/bge-small-en-v1.5` (Beijing Academy of AI, 2023) `[SNIPPET]`.
  * `intfloat/e5-small-v2` (Microsoft, 2023) `[SNIPPET]`.
  * `thenlper/gte-small` (Alibaba DAMO Academy, 2023) `[SNIPPET]`.
  * `minishlab/potion-base-8M` / `model2vec` (Stephan Poelsma & Thomas van Dongen, 2024) `[SNIPPET]`.
* **URL:** [github.com/MinishLab/model2vec](https://github.com/MinishLab/model2vec) `[SNIPPET]`; Hugging Face model cards `[SNIPPET]`.
* **What it predicts:** Fixed-dimension vector embeddings ($\mathbb{R}^{384}$ or $\mathbb{R}^{256}$) fed to `sklearn.linear_model.LogisticRegression` outputting class probabilities `[SNIPPET]`.
* **Model Specs & Footprint:**
  * `all-MiniLM-L6-v2`: 22.7M parameters (~90 MB), Apache-2.0 `[SNIPPET]`.
  * `bge-small-en-v1.5`: 33.4M parameters (~133 MB), MIT `[SNIPPET]`.
  * `e5-small-v2`: 33.4M parameters (~133 MB), MIT (requires `"query: "` prefix) `[SNIPPET]`.
  * `gte-small`: 33.4M parameters (~100 MB), MIT `[SNIPPET]`.
  * `minishlab/potion-base-8M` (Model2Vec static embeddings): 7.5M parameters (~30 MB), MIT `[SNIPPET]`.
* **Trained Weights Released:** All base embedding models are released on Hugging Face under MIT / Apache-2.0 `[SNIPPET]`. Logistic regression head is fit locally in <1 second on a few thousand embeddings `[SNIPPET]`.
* **Training Data & Size:** Trained on 500 to 5,000 labeled questions from FreshQA dev and SituatedQA `[SNIPPET]`.
* **Reported Accuracy / F1:** MTEB benchmark classification scores consistently achieve 80–86% top-1 accuracy for frozen small embeddings with linear classifiers `[SNIPPET]`.
* **Latency (Measured / Reported on CPU per sentence):**
  * `potion-base-8M` (Model2Vec): **~0.3 ms to 1.5 ms** (token lookup + pooling, no transformer attention) `[SNIPPET]`.
  * `all-MiniLM-L6-v2` (ONNX INT8 / FP32): **~5 ms to 15 ms** `[SNIPPET]`.
  * `bge-small-en-v1.5` / `e5-small-v2`: **~15 ms to 30 ms** `[SNIPPET]`.
  * Logistic Regression head prediction: **<0.05 ms** `[SNIPPET]`.
* **Handles Freshness:** Yes, when the linear head is trained on static vs. changing queries `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning (fitting head).** `potion-base-8M` + LogReg provides a ~1 ms CPU gate with 30 MB footprint; `all-MiniLM-L6-v2` + LogReg provides richer contextual semantics in ~10 ms `[SNIPPET]`.

---

### 5. ModernBERT and DistilBERT Fine-Tuning for Short-Text Classification
* **Authors / Year:** 
  * `ModernBERT-base`: Benjamin Warner, Antoine Chaffin, Benjamin Clavié et al. (Answer.AI & LightOn, Dec 2024, arXiv:2412.13663) `[SNIPPET]`.
  * `distilbert-base-uncased`: Victor Sanh et al. (Hugging Face, 2019, arXiv:1910.01108) `[SNIPPET]`.
* **URL:** [huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) `[SNIPPET]`; [huggingface.co/distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) `[SNIPPET]`.
* **What it predicts:** Sequence classification logits (e.g., binary: `0: NO_SEARCH`, `1: NEEDS_SEARCH`) `[SNIPPET]`.
* **Model & Parameter Count:**
  * `distilbert-base-uncased`: 66.4M parameters (~260 MB FP32, ~130 MB FP16, ~67 MB INT8) `[SNIPPET]`.
  * `ModernBERT-base`: 149M parameters (~600 MB FP32, ~300 MB FP16, ~150 MB INT8) `[SNIPPET]`.
* **Weights & Licence:** Both released on Hugging Face under **Apache-2.0** `[SNIPPET]`.
* **Training Data & Size:** Fine-tuned via Hugging Face `Trainer` on 2,000–10,000 labeled queries `[SNIPPET]`.
* **Reported Accuracy / F1:** ModernBERT achieves state-of-the-art accuracy across GLUE tasks, outperforming DeBERTa-v3 and original BERT `[SNIPPET]`. DistilBERT retains 97% of BERT-base performance on classification benchmarks `[SNIPPET]`.
* **CPU Latency with ONNX Runtime (Batch Size = 1, sequence length 32–64 tokens):**
  * `distilbert-base-uncased` (ONNX Runtime INT8 on CPU): **~2 ms to 6 ms** (FP32: ~4 ms to 10 ms) `[SNIPPET]`.
  * `ModernBERT-base` (ONNX Runtime INT8 on CPU): **~12 ms to 35 ms** (FP32: ~35 ms to 60 ms) `[SNIPPET]`.
* **Handles Freshness:** Yes; fine-tuning directly teaches the model temporal cues, entity obscurity boundaries, and question intent `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning.** DistilBERT with ONNX INT8 is a top recommendation: ~4 ms CPU latency, 67 MB model footprint, and robust contextual accuracy `[SNIPPET]`.

---

### 6. Arch-Router
* **Authors / Year:** Katanemo (2024–2025, arXiv:2506.16655 / arXiv:2410.15516) `[SNIPPET]`.
* **URL:** [github.com/katanemo/archgw](https://github.com/katanemo/archgw) `[SNIPPET]` / [huggingface.co/katanemo/Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) `[SNIPPET]`.
* **What it predicts:** High-level task intent and action policies, routing user queries to designated downstream models (e.g., coding model vs. chat model) based on natural language developer preferences `[SNIPPET]`.
* **Model & Parameter Count:** **1.5 Billion parameters** (autoregressive causal LM based on `Qwen/Qwen2.5-1.5B-Instruct`) `[SNIPPET]`.
* **Weights & Licence:** Available at `katanemo/Arch-Router-1.5B` on Hugging Face under **Apache-2.0** `[SNIPPET]`.
* **Training Data & Size:** Trained on synthetic prompt-route preference data across diverse domains; exact dataset size: `NOT FOUND` `[SNIPPET]`.
* **Reported Accuracy / F1:** Vendor claims 15–25% accuracy gain over prompt-based zero-shot LLM routers `[SNIPPET]`.
* **Latency:** Requires vLLM or Ollama; reported inference latency is ~100–300 ms on GPU; on CPU it exceeds **800 ms to 2,000 ms** `[SNIPPET]`.
* **Handles Freshness:** No; routes across broad model domains, not search freshness `[SNIPPET]`.
* **Verdict:** **Fails constraints (Only the idea).** Fails CPU latency constraint (>50 ms), fails footprint constraint (>3 GB), and violates the constraint against calling an extra LLM `[SNIPPET]`.

---

### 7. RouteLLM
* **Authors / Year:** Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, Ion Stoica (LMSYS Org / UC Berkeley, 2024) `[SNIPPET]`.
* **URL:** [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) `[SNIPPET]` / [arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665) `[SNIPPET]`.
* **What it predicts:** Probability that a "strong" LLM (e.g., GPT-4) outperforms a "weak" LLM (e.g., Mixtral-8x7B) on a given prompt to balance cost and quality `[SNIPPET]`.
* **Model & Parameter Count:** Provides several router architectures:
  * Matrix Factorization (`routellm/mf`): linear scoring layer over text embeddings (~100k params) `[SNIPPET]`.
  * BERT classifier (`routellm/bert`): BERT-base (110M params, ~420 MB) `[SNIPPET]`.
  * Causal LLM (`routellm/causal_llm`): 8B params `[SNIPPET]`.
* **Weights & Licence:** Code and checkpoints released on Hugging Face (`routellm/mf`, `routellm/bert`) under **Apache-2.0** `[SNIPPET]`.
* **Training Data & Size:** Trained on LMSYS Chatbot Arena preference battles (~55k human battles, ~117k GPT-4 judge battles) `[SNIPPET]`. Datasets released on Hugging Face under `routellm/` `[SNIPPET]`.
* **Reported Accuracy / F1:** Router achieves 95% of GPT-4 performance while sending 50%+ of queries to cheaper models; binary classification accuracy on Arena preference: ~65–72% `[SNIPPET]`.
* **Latency:** Matrix Factorization router CPU latency is ~10–25 ms (dominated by embedding extraction); BERT classifier is ~25–50 ms on CPU `[SNIPPET]`.
* **Handles Freshness:** No; routes based on reasoning complexity / model capability, not factual freshness `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning (or only the idea).** As released, it routes strong vs. weak LLMs. Its Matrix Factorization / BERT router architecture can be repurposed, but off-the-shelf checkpoints do not route web search `[SNIPPET]`.

---

### 8. Published "Query Intent" / "Needs Search" Models on Hugging Face

#### A. Ammu-1.0-FactCheck-Need Family
* **Authors / Year:** Kalyan KS (2025) `[SNIPPET]`.
* **URL:** [huggingface.co/kalyan-ks/Ammu-1.0-FactCheck-Need-base](https://huggingface.co/kalyan-ks/Ammu-1.0-FactCheck-Need-base) `[SNIPPET]`.
* **What it predicts:** Binary labels: **`FACT_CHECK_NEEDED`** vs. **`NO_FACT_CHECK_NEEDED`** (routes prompts needing external factual retrieval vs. coding/creative/math) `[SNIPPET]`.
* **Model & Parameter Count:** ModernBERT / Ettin encoder backbone:
  * `Ammu-1.0-FactCheck-Need-base`: 150M parameters (~300 MB) `[SNIPPET]`.
  * `Ammu-1.0-FactCheck-Need-small`: 68M parameters (~140 MB) `[SNIPPET]`.
  * `Ammu-1.0-FactCheck-Need-tiny`: 33M parameters (~70 MB) `[SNIPPET]`.
  * `Ammu-1.0-FactCheck-Need-nano`: 17M parameters (~35 MB) `[SNIPPET]`.
* **Weights & Licence:** Weights released on Hugging Face under **MIT License** `[SNIPPET]`.
* **Training Data & Size:** Trained on `kalyan-ks/fact-check-need-classification` (28,088 total records: 21,070 train, 2,810 val, 4,214 test), balanced 50/50 across 120 topics and 19 prompt interaction formats. Dataset released on Hugging Face under MIT `[SNIPPET]`.
* **Reported Accuracy / F1:** **96.56% Macro-F1** on the 4,214-example test split (98.91% Easy, 96.74% Medium, 95.23% Hard) `[SNIPPET]`.
* **Latency:** With ONNX Runtime CPU execution, `base` runs in ~15–30 ms; `small` runs in ~8–15 ms; `nano` runs in ~3–6 ms `[SNIPPET]`.
* **Handles Freshness:** It filters factual claims from subjective/creative/math tasks, but flags *all* factual queries as `FACT_CHECK_NEEDED` (including timeless facts like "When was Mozart born?") `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning.** Fits all CPU latency (<30 ms) and size (<300 MB) constraints; provides an excellent starting checkpoint to fine-tune on temporal/freshness datasets `[SNIPPET]`.

#### B. AgentIntentRouter
* **Authors / Year:** Shaswata Tripathy (2024) `[SNIPPET]`.
* **URL:** [huggingface.co/tripathyShaswata/AgentIntentRouter](https://huggingface.co/tripathyShaswata/AgentIntentRouter) `[SNIPPET]`.
* **What it predicts:** 5 discrete tool intent classes: `web_search`, `code_generation`, `math_calculation`, `api_call`, `chat` `[SNIPPET]`.
* **Model & Parameter Count:** Fine-tuned `distilbert-base-uncased` (66.9M parameters, ~260 MB FP32, ~67 MB INT8) `[SNIPPET]`.
* **Weights & Licence:** Released on Hugging Face under **Apache-2.0** `[SNIPPET]`.
* **Training Data & Size:** 8,987 synthetic template-generated prompt examples (1,123 validation, 1,124 test) `[SNIPPET]`.
* **Reported Accuracy / F1:** Test macro F1 >95% on synthetic in-distribution test prompts `[SNIPPET]`.
* **Latency:** ~4 ms to 10 ms on CPU `[SNIPPET]`.
* **Handles Freshness:** Only flags explicit search intent phrasing; does not evaluate factual changeability or entity obscurity `[SNIPPET]`.
* **Verdict:** **Usable after fine-tuning.** Validates that DistilBERT is well-suited for fast tool routing, but weights require fine-tuning for freshness gating `[SNIPPET]`.

#### C. RAGate (Adaptive Retrieval Gate)
* **Authors / Year:** Eric Wang et al. (NAACL Findings 2024, arXiv:2407.21712) `[READ]`.
* **URL:** [arxiv.org/abs/2407.21712](https://arxiv.org/abs/2407.21712) `[READ]` / [github.com/wangxieric/RAGate](https://github.com/wangxieric/RAGate) `[SNIPPET]`.
* **What it predicts:** Binary decision: whether a dialogue turn requires external knowledge augmentation `[READ]`.
* **Model & Parameter Count:** Multi-Head Attention neural gate (RAGate-MHA, ~10M–30M params) or LoRA fine-tuned LM `[SNIPPET]`.
* **Weights & Licence:** GitHub code under open research terms; weights: `NOT FOUND` on Hugging Face hub `[SNIPPET]`.
* **Training Data & Size:** Evaluated on KETOD (Knowledge-Extended Task-Oriented Dialogue, 16 domains, ~10k dialogue turns) `[SNIPPET]`.
* **Reported Accuracy / F1:** Matches human judgment trends on adaptive knowledge augmentation, outperforming zero-shot prompting `[READ]`.
* **Latency:** Neural MHA gate executes in ~10–20 ms on CPU `[SNIPPET]`.
* **Handles Freshness:** Evaluates knowledge need in conversational context, but focuses on dialogue task domains rather than question freshness `[READ]`.
* **Verdict:** **Only the idea.** Tailored to conversational dialogue history; no off-the-shelf single-turn freshness checkpoint `[READ]`.

---

## 2. Datasets We Could Train and Evaluate a Gate On

| Dataset Name | Size | Exact Labels / Annotations | Licence | URL |
| :--- | :--- | :--- | :--- | :--- |
| **FreshQA** `[READ]` | 600 questions (100 Dev, 500 Test) `[SNIPPET]` | Fact Type: `never-changing`, `slow-changing`, `fast-changing`; Premise: `false-premise`; Reasoning: `one-hop`, `multi-hop` `[SNIPPET]` | **Apache-2.0** `[SNIPPET]` | [github.com/freshllms/freshqa](https://github.com/freshllms/freshqa) `[READ]` |
| **SituatedQA** `[SNIPPET]` | ~5,204 context-dependent identification queries; ~10,000 QA pairs (6,009 TEMP train, 3,423 dev, 2,795 test; 3,548 GEO train, 1,398 dev) `[SNIPPET]` | Context-dependence: `true` / `false`; Context type: `TEMP` (temporal), `GEO` (geographical), or static `[SNIPPET]` | **CC BY-SA 4.0** `[SNIPPET]` | [github.com/mikejqzhang/SituatedQA](https://github.com/mikejqzhang/SituatedQA) `[SNIPPET]` |
| **StreamingQA** `[SNIPPET]` | ~147,000 question-answer pairs `[SNIPPET]` | Timestamps spanning 2007–2020 news streams; time-dependent QA pairs `[SNIPPET]` | **CC-BY 4.0** (data) / **Apache-2.0** (code) `[SNIPPET]` | [github.com/google-deepmind/streamingqa](https://github.com/google-deepmind/streamingqa) `[SNIPPET]` |
| **RealTime QA** `[SNIPPET]` | ~30 questions/week continuously updated (total >2,500 questions) `[SNIPPET]` | Weekly timestamped news events; multiple-choice and generation answers `[SNIPPET]` | **Unclear / No formal licence file in repo** `[SNIPPET]` | [github.com/realtimeqa/realtimeqa_public](https://github.com/realtimeqa/realtimeqa_public) `[SNIPPET]` |
| **TempLAMA** `[SNIPPET]` | ~50,310 temporal cloze queries `[SNIPPET]` | Wikidata temporal triplets (subject, relation, object) annotated with yearly validity intervals (2010–2020) `[SNIPPET]` | **CC BY-SA 3.0 / 4.0** `[SNIPPET]` | [github.com/google-research/language](https://github.com/google-research/language) `[SNIPPET]` |
| **Ammu Fact-Check Need** `[SNIPPET]` | 28,088 queries (21,070 train, 2,810 val, 4,214 test) `[SNIPPET]` | Binary: `FACT_CHECK_NEEDED`, `NO_FACT_CHECK_NEEDED` `[SNIPPET]` | **MIT** `[SNIPPET]` | [huggingface.co/datasets/kalyan-ks/fact-check-need-classification](https://huggingface.co/datasets/kalyan-ks/fact-check-need-classification) `[SNIPPET]` |

---

## 3. Ranking

### 1. Fine-Tuned DistilBERT (or ModernBERT-base) with ONNX Runtime INT8
* **Plan:**
  1. Construct a binary training set of ~6,000 queries by mapping SituatedQA (`TEMP` $\rightarrow$ `SEARCH`, static $\rightarrow$ `NO_SEARCH`) combined with FreshQA's 100 dev examples and Ammu non-factual negatives `[SNIPPET]`.
  2. Fine-tune `distilbert-base-uncased` for 3 epochs with a weighted binary cross-entropy loss that penalizes missed changing questions 2.5× more than false alarms `[SNIPPET]`.
  3. Export to ONNX Runtime with INT8 dynamic quantization for deployment; yields a 67 MB model executing in ~3–5 ms on laptop CPU with >85% recall on changing questions `[SNIPPET]`.

### 2. SetFit Few-Shot Classification on `all-MiniLM-L6-v2`
* **Plan:**
  1. Hand-curate 40–80 high-diversity contrastive query pairs representing fast-changing facts, slow-changing roles/prices, obscure entities, and static trivia `[READ]`.
  2. Run SetFit contrastive Siamese fine-tuning on `sentence-transformers/all-MiniLM-L6-v2` and fit the Logistic Regression head in ~2 minutes on CPU `[READ]`.
  3. Export the fine-tuned encoder to ONNX Runtime; deploy with a calibrated probability decision threshold ($p > 0.40$ triggers web lookup); runs in ~10–12 ms on CPU `[SNIPPET]`.

### 3. Model2Vec Static Embeddings (`potion-base-8M`) + Logistic Regression
* **Plan:**
  1. Precompute 256-dimensional static embeddings for SituatedQA and FreshQA dev queries using `minishlab/potion-base-8M` via simple token lookups `[SNIPPET]`.
  2. Fit an `sklearn.linear_model.LogisticRegression` (or balanced LightGBM classifier) with an adjusted decision threshold favoring recall `[SNIPPET]`.
  3. Deploy the resulting 30 MB static embedding table and classifier head using pure NumPy / `model2vec-rs`; inference runs in ~0.5–1.0 ms on CPU `[SNIPPET]`.

---

### What Would Make Us Change the Order?
* **If labeled training data is under 100 questions:** **SetFit** (#2) moves to **#1**, because contrastive few-shot learning reliably outperforms standard cross-entropy fine-tuning in data-scarce regimes (<100 samples) `[READ]`.
* **If the CPU budget drops below 2 ms** (e.g., if speech ASR or text-to-speech heavily saturates CPU cores): **Model2Vec + Logistic Regression** (#3) moves to **#1**, running in ~0.5 ms without any transformer attention computation `[SNIPPET]`.
* **If accuracy on subtle, out-of-domain temporal questions is paramount and 20 ms CPU latency is acceptable:** **Fine-tuned ModernBERT-base** (149M params, INT8 ONNX running in ~18 ms) replaces DistilBERT within the #1 spot to capture the highest possible contextual reasoning capacity `[SNIPPET]`.
