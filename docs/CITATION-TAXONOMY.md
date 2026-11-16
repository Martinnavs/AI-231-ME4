# Which questions need a source: existing taxonomies and what they change for us (2026-11-16)

Three agy runs (plan mode, search and page reads only) on freshness, attribution and deployed systems. Raw reports: `.scratch/citation-taxonomy/AGY-{A-freshness,B-attribution,C-industry}.md` (untracked). Agy's evidence is weaker than codex's (docs/OPTIMIZATION-RESEARCH.md): below, **checked** means I read the page myself on 2026-11-16, **agy only** means the claim rests on its report (a page read or a search snippet it marked).

## Taxonomies that exist

| Source | Categories | Status |
|---|---|---|
| FreshQA (Vu et al. 2023, arXiv 2310.03214) | never-changing, slow-changing, fast-changing, false-premise | **checked** (paper text); its abstract: all models struggle on fast-changing and false-premise questions |
| SituatedQA (Zhang and Choi 2021, 2109.06157) | static, temporal (TEMP), geographic (GEO); agy: 16.5% of Natural Questions are context-dependent | agy only |
| Adaptive-RAG (Jeong et al. 2024, 2403.14403) | non-retrieval, single-step, multi-step, picked by a small trained classifier | classifier **checked** (abstract); the three tier names agy only |
| Self-RAG (2310.11511) | learned [Retrieve] decision per segment, with a probability threshold | agy only; needs fine-tuning, so not usable at our latency |
| Mallen et al. 2023 (2212.10511) | popularity: head, torso, tail; memory fails on the tail even for unchanging facts | agy only |
| AIS (2112.12870), ALCE (2305.14627) | attributable vs not; citation recall and precision | agy only |
| FActScore, SAFE | atomic claims, supported or not | agy only; an offline evaluation, not a gate |
| ClaimBuster, CLEF CheckThat! | non-factual, unimportant factual, check-worthy factual | agy only |
| Wikipedia WP:V | cite what is "challenged or likely to be challenged", all quotations, contentious material about living people; common knowledge exempt | agy only (snippet) |
| Google YMYL (rater guidelines) | health and safety, financial, civics/government/law, news; cite always | agy only (snippet) |
| Google QDF (2011 blog) | recent events, regularly recurring events, frequent updates | agy only (snippet) |
| Gemini grounding | the model decides; `dynamic_threshold` existed for Gemini 1.5/2.0 | **checked**: the current page (aistudio.google.com/docs/google-search, moved from ai.google.dev) says only that the model "analyzes the prompt and determines if a Google Search can improve the answer"; no threshold is documented. Agy's claim that Gemini 3.x still has `dynamic_threshold` is **not supported** |

Likely wrong in the agy reports: the "8-class Redi et al. taxonomy" (Missing Information, Medical Content, POV, Doubt, Vague, Jargon, Unclear, Sic) reads like Wikipedia cleanup tags, not that paper's citation reasons; not verified. The spoken-attribution rules in report C ("front-load", "one source only", "hedging penalty", "rule of one fact") rest on search snippets behind redirect URLs: treat as hypotheses.

## What it means for our one regex

Our `changeable()` is FreshQA's fast and slow classes found by keyword. The literature names what it misses:

1. **Implicit temporal questions** ("How many seasons does X have?"): agy cites 7-17% of queries as implicitly temporal (snippets); a role-and-"latest" regex cannot see them.
2. **Geographic dependence** (SituatedQA GEO: "here", "near me", local rules, "the emergency number"): we have the venue, but no gate.
3. **Long-tail entities** (Mallen): an unchanging fact about an obscure entity is still answered wrongly from memory. Our grounding prompt says "settled facts (capitals, authors, historical dates) need no search", which is exactly the case the paper warns about.
4. **High stakes** (YMYL: medicine, law, money, civics): cite always, changeable or not.
5. **False premises** (FreshQA): the model accepts them; a prompt line costs nothing.
6. **Quotes and contested claims** (WP:V).

## Proposed policy (a decision for the owner)

Smallest change that matches the professor's rule ("grounded every time"): search and cite for **every factual information request**, with a short never-list: arithmetic, definitions of common words, conversation, creative writing, home commands, questions about the conversation itself. That drops the need to predict what changes. Cost: a settled question moves from about 0.9 s (no search) to about 1.8 s (one search), from the ledger's medians.

If that latency is not acceptable, keep the gate but widen it to: fast/slow (now), geographic (regex: here, near me, local, nearest), YMYL (lexicon), and an unknown-entity cue (a proper noun not in the head list), plus the false-premise prompt line.

Not worth adopting at voice latency: Self-RAG, FLARE and any multi-pass drafting, FActScore/SAFE online, bracketed numeric citations, a `dynamic_threshold` knob (not documented for Gemini 3).

## Open questions

- How often do real turns fall in each class? The turn log's `unchecked` flag and the ledger's `query` column make this countable after a few sessions.
- Front-loaded attribution ("According to Nvidia, ...") versus our trailing "Source: Nvidia.": untested; the evidence for front-loading is a snippet.
- Whether the false-premise line changes answers: needs FreshQA's false-premise split run against the agent (cost: a dry run).

## Adaptive-RAG's classifier: can we integrate it? (checked 2026-11-16)

Read from the paper's HTML (arXiv 2403.14403) and the repo README (github.com/starsuzi/Adaptive-RAG, Apache-2.0):

- **What it is:** a T5-Large (770M) that labels a query A (answer without retrieval), B (single-step retrieval), C (multi-step). The paper reports no significant difference across classifier sizes.
- **How it is trained:** silver labels from which strategy answered correctly on benchmark queries (the simplest strategy that did wins), plus "B for single-hop datasets, C for multi-hop datasets"; 400 queries from each of 6 datasets.
- **How good it is (their Table 4):** 54.52% overall; 30.52% on A, 66.28% on B, 65.45% on C. It is weakest on the class that skips retrieval, which is the class we cannot afford to get wrong.
- **Released?** The README links data (`data.tar.gz`) and training scripts, no trained checkpoint; the silver labels need LLM-plus-retrieval runs we would have to pay for.

**It cannot be dropped in.** It measures reasoning depth (how many hops), not how fast an answer goes stale, and its training questions are static Wikipedia benchmarks, so it has never seen a changing fact; it also exists to save retrieval, where we want to retrieve every time. A 770M model on the Pi would break the "no model on the Pi except Silero" rule, and the agent server runs on the Pi in production (`HARNESS=pi`). Its C class maps to our `web_research` or delegate path, which we route by prompt already.

**What is worth taking is the method:** a small classifier trained on silver labels taken from outcomes. For us the outcome is cheap to get: ask Gemini the same question with and without a search and label the question "needs a lookup" when the two answers differ. The ledger now logs the query and the search count, so real traffic can supply the questions.

**How the current gate does (new: `scripts/freshqa_gate_check.py`, FreshQA's April 2026 sheet, 600 questions, no model calls):**

| | n | flagged by `changeable()` |
|---|---|---|
| never-changing | 225 | 21% (false alarms) |
| slow-changing | 220 | 40% |
| fast-changing | 155 | 77% |
| changing with a valid premise | 301 | 63% (the share we must catch) |

Missed examples: "Who is the richest man on earth?", "How many novels has Stephen King published?", "Can women attend universities in Afghanistan?". As a yardstick, a bag-of-words logistic regression under 5-fold cross-validation reaches 0.86 recall on changing questions at 0.36 false alarms, against the regex's 0.55 at 0.21 (labels: changing = slow or fast, either premise). That is in-distribution on 600 questions from one team of writers, so read it as "a learned gate has room to beat the regex", not as a number for our traffic.

**Verdict:** do not integrate the Adaptive-RAG classifier. Either drop the gate (option A above: search every factual question, which removes the miss rate) or train our own small gate on outcome labels, served where the router already runs (the laptop), never on the Pi.

## Similar classifiers (survey of 2026-11-16; reports in `.scratch/classifier-survey/AGY-*.md`)

Three more agy runs, then I read the key model cards and **ran the two released candidates myself** on FreshQA's 600 questions (CPU, one thread, PyTorch fp32 in a scratch venv; scripts `.scratch/classifier-survey/{eg_eval,ammu_eval}.py`).

**Released and usable: EG-E5, a classifier for exactly our question.** `s-nlp/E5-EverGreen-Multilingual-Small` (Pletenev et al. 2025, arXiv 2505.21115): predicts Mutable or Evergreen from the question text alone; 118M parameters (471 MB, safetensors), MIT, fine-tuned from `intfloat/multilingual-e5-small` on EverGreenQA (`s-nlp/EverGreen-Multilingual`: train 2,010 + synthetic 1,480, test 1,270, seven languages; label 0 = mutable, 1 = evergreen). Its card reports 0.88 F1 on English (agy said 0.92 and 0.88 accuracy; the card is the number to trust). Zero-shot on FreshQA, never seeing it:

| gate | recall, changing | false alarms, never-changing | fast | slow |
|---|---|---|---|---|
| our regex | 0.55 | 0.21 | 0.77 | 0.40 |
| EG-E5, P(mutable) >= 0.5 | **0.78** | **0.17** | **0.90** | **0.70** |
| regex OR EG-E5 | 0.83 | 0.31 | 0.95 | 0.75 |

AUC 0.87, 19 ms median (p90 23 ms) per question on one CPU thread. It beats the regex on every axis with no training. It still misses obscure-entity questions ("When was Dartmouth Law School founded?", "When was League of Legends 2 released?") and false-alarms on counts and durations ("How many episodes of ... season 1 have been released?"). Unverified: whether EverGreenQA overlaps FreshQA (the card does not say), and the dataset's licence (the card page shows none; agy says MIT).

**Released, a different question: Ammu FactCheck-Need** (`kalyan-ks/Ammu-1.0-FactCheck-Need-small`, MIT, 68M, Ettin encoder): does a prompt contain facts to verify. It is not a freshness gate: it flagged 92% of never-changing, 97% of slow- and 99% of fast-changing FreshQA questions. But it separates factual requests from math, jokes, commands, how-to and creative prompts (7 of 7 non-factual probes passed; it wrongly flagged "Thanks, that's all."). That is the never-list for option A (search every factual request). 38 ms per question here (the pipeline wrapper included).

**Not usable as released (ideas only):** UAR, Probing-RAG, SeaKR, DRAGIN, KBM need the LLM's hidden states or logits (we call hosted Gemini); RA-ISF, Rowen, Self-DC call an LLM several times; SKR, RAGate, the SituatedQA classifier (BERT-large, 88.5% on TEMP in the paper) and CRAG's dynamism router (bge-m3, 567M) released no weights; Arch-Router (1.5B) and RouteLLM route between models, not search. Agy's latency and licence figures for these are snippets.

**Datasets to train or evaluate on:** FreshQA (600, Apache-2.0, checked), EverGreenQA (above), SituatedQA TEMP (about 4.4k train, CC BY-SA 4.0 per agy; HF `siyue/SituatedQA`), CRAG (4,409 with dynamism labels, CC BY-NC per agy), UAR's AR-Bench time-aware subset (about 8k, licence unclear), PopQA (popularity, for the obscure-entity gap).

**Where it can run (the constraint that decides the design).** The agent server and its MCP children run on the Pi in production (`HARNESS=pi`), and the rule is no model on the Pi except Silero. So the gate must be a service on the laptop, next to the router, called with a short timeout and the regex as fallback. In text mode (everything on the laptop) it can run in process. The root venv already has `onnxruntime` and `tokenizers`, so exporting EG-E5 to ONNX once (int8, vocabulary trimmed: the 250k-token embedding is most of the 471 MB) avoids adding torch to the repo.

**Done (2026-11-16, `text-only`):** exported to ONNX and quantized to int8 by `scripts/export_gate_onnx.sh` (`make setup-gate`, 30 s once the downloads are cached): 118 MB, 4.2 ms median on one thread (fp32 ONNX 9.2 ms, identical to PyTorch), FreshQA 600: recall 0.80, false alarms 0.18 (never-changing 18%, slow 71%, fast 92% flagged). `voice_agent.gate` loads it lazily with `onnxruntime` and `tokenizers`; `changeable()` uses it and falls back to the regex without the model files or libraries (the Pi has neither). `tests/test_gate.py` asserts the numbers on a 120-question fixture. **Not yet decided or tested:** whether fine-tuning EG-E5 on FreshQA dev plus SituatedQA closes the obscure-entity gap; combining it with Ammu and the regex (the table's OR row costs 0.14 more false alarms for 0.05 more recall).
