# Small Reasoning & Cheap LLM Research for ME4 (2026-11-27)

Scope: public-web evaluation of 2026 OpenRouter models for ME4's three small LLM jobs (**1. Judge**, **2. Splitter**, **3. Short Reasoning**). No live API calls or paid tests were run here; local references (`Qwen3.8-27B` lab server, `Gemini 3.1 Flash-Lite`) come from repo docs. Claims older than `2026-06` are marked `[stale <2026-06]`; unverified catalog/provider flags are marked `unverified`.

## A. Shortlist (OpenRouter, discovered 2026-11-27)

| OpenRouter ID | Type | $/M In / Out | Ctx | Logprobs | Struct Out | Reasoning Controls | Speed (TTFT / tok/s) | ZDR / Privacy Notes |
|---|---|---:|---:|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | Prop (Oct 2026) | $0.10 / $0.50 (<=100K) | 1M | Unlikely (`unverified`; Anthropic API omits) | Yes (`json_schema` / strict tools) | Mandatory adaptive thinking (`effort`: `low`..`max`; `enabled:false` -> 400; `temp`/`top_p` ignored) | ~1.1–1.4s TTFT; 154–382+ tok/s | US-only/ZDR routes available (1.1x US-only multiplier) |
| `google/gemini-3.5-flash-lite` | Prop (Jul 2026) | $0.30 / $2.50 ($0.03 cache) | 1.05M | Conditional (`unverified` per OR route) | Yes (native `json_schema`) | `thinking_level` (`minimal`/`low`/`medium`/`high`) | TTFT `unverified`; ~350 tok/s (AA) | Paid/Vertex APIs do not train on prompts; ZDR via Vertex |
| `openai/gpt-5-nano` | Prop (GPT-5) | $0.05 / $0.40 ($0.025/$0.20 batch) | 400K | **No** (OpenAI reasoning models return 400) | Yes (strict JSON schema) | `effort` (`minimal`/`low`) | Fastest GPT-5 tier; ~80–110+ tok/s (`p50 unverified`) | ZDR on OpenAI/Azure enterprise endpoints |
| `openai/gpt-5-mini` | Prop (GPT-5) | $0.25 / $2.00 | 400K | **No** (OpenAI reasoning models return 400) | Yes (strict JSON schema) | `effort` (`low`/`medium`/`high`/`xhigh`) | ~60–120+ tok/s class (`p50 unverified`) | ZDR on OpenAI/Azure enterprise endpoints |
| `deepseek/deepseek-v4-flash-0731` | Open (Jul 2026) | ~$0.045 / $0.136–$0.30 | 1.05M | Provider-dep (`unverified` per host) | Yes (provider-dep) | Native CoT (`effort` / `max_tokens` / `exclude`) | ~0.38–0.85s latency; 100–168 tok/s | Must enforce `provider.zdr: true` (multi-host resellers) |
| `qwen/qwen3.8-flash` | Open/Cloud (Aug 2026) | $0.15 / $0.47 | `unverified` (131K–1M line) | Provider-dep (`unverified` on Flash) | Yes (OpenAI-compat) | `reasoning` (`enabled` / `max_tokens` / `exclude`) | ~0.1–0.25s TTFT; 70–150+ tok/s class | Must enforce `provider.zdr: true` |
| `qwen/qwen3.8-27b` | Open (Aug 2026) | ~$0.035–$0.16 / $1.50–$2.00 | 131K–262K (`unverified`) | **Yes** (lab server + open-weight OR hosts) | Yes (OpenAI-compat) | Toggleable thinking (`enabled` / `max_tokens`) | 70–150+ tok/s hosted; ~2.5s med on lab | Must enforce `provider.zdr: true` on OR; local on lab |
| `mistralai/mistral-small-2603` | Open 119B MoE (Mar 2026) | $0.15 / $0.60 ($0.015 cache) | 256K–262K | **Yes** (compatible OR hosts) | Yes (native) | Configurable reasoning effort (unified Magistral) | **0.38s** p50; **136 tok/s** (OR) | Must enforce `provider.zdr: true` |
| `google/gemma-4-26b-a4b-it` | Open MoE 3.8B act (Apr 2026) | ~$0.07–$0.09 / $0.22–$0.30 (`:free` tier) | 262K | **Yes** (compatible OR hosts) | Yes (native) | Toggleable `enable_thinking` / effort | 50–130+ tok/s (`p50 unverified`) | Must enforce `provider.zdr: true`; avoid `:free` for PII |
| `z-ai/glm-5.3-flash` | Open/Cloud (2026) | $0.036 / $0.082 | `unverified` (1M on `glm-5.3`) | Provider-dep (`unverified`) | `unverified` | Native reasoning (`high`/`xhigh` on GLM-5.3) | `unverified` (Flash tier) | Provider-dependent; enforce `provider.zdr: true` |

Notes: Catalog specs live at `https://openrouter.ai/api/v1/models` (`supported_parameters` lists `logprobs`, `structured_outputs`, `reasoning` per endpoint). Prior `google/gemini-3.1-flash-lite-preview` ($0.25/$1.50, 362–389 tok/s) is superseded by `3.5-flash-lite`; `minimax/minimax-m2.7` ($0.60/$1.80, 198K ctx, 0.59s p50, 183.5 tok/s) is also fast but pricier than Haiku 5.5 / Mistral Small 4.

## B. Evidence of Quality for ME4 Jobs (not generic leaderboards)

| Benchmark / Source | Date | Type | Key Finding for ME4 Jobs | Link |
|---|---|---|---|---|
| Reliability without Validity (Judge Audit) | 2026-06 | Empirical study | "Consistency-Bias Paradox": judges hit >0.95 test-retest stability while hiding >0.10 position bias; swap/single-turn checks required. | [arXiv:2606.19544](https://arxiv.org/abs/2606.19544) |
| Auditing the Judge / Position-Stable Agreement | 2025–2026 | Study (`[stale <2026-06]` parts) | Style (bullet lists, tone, jargon) flips verdicts as much as raw length; minimalist prompts can collapse 70–100% to one slot. | [ACL Anthology](https://aclanthology.org) / [arXiv](https://arxiv.org) |
| JudgeBench & RewardBench 2 | 2024-10 / 2025-06 | Benchmarks `[stale <2026-06]` | Hard factual/reasoning splits separate real verification from style; GPT-4.1 (72.3) & Claude Sonnet (71.2) trail tuned 4B–70B RMs (70–74). | [arXiv:2410.12784](https://arxiv.org/abs/2410.12784), [arXiv:2506.01937](https://arxiv.org/abs/2506.01937) |
| FACTS Grounding & LLM-AggreFact | 2025-01 / 2024-04 | Grounding `[stale <2026-06]` | Grounded faithfulness favors strict entailment checkers; GPT-5 hits ~61.8% FACTS (low contradiction, higher hedging); Flash 3.7/3.8 ~66–75.8% on extensions. | [arXiv:2501.03200](https://arxiv.org/abs/2501.03200), [arXiv:2404.10774](https://arxiv.org/abs/2404.10774) |
| JSONSchemaBench & IFEval-FC | 2025-01 | Struct Out `[stale <2026-06]` | Constrained decoding (`response_format: json_schema` / XGrammar) matters more than model size for syntax; value accuracy remains the gap. | [arXiv:2501.10868](https://arxiv.org/abs/2501.10868) |
| Anthropic Claude Haiku 5.5 Release & Evals | 2026-10-07 | Vendor + 3rd-party evals | 89.6% GPQA Diamond (max effort), AA Index ~43, but **23.8% SimpleQA Verified**—strong reasoning/routing, weak parametric world knowledge. | [Anthropic](https://www.anthropic.com/news/claude-haiku-5-5), [OpenRouter](https://openrouter.ai) |
| Google Gemini 3.5 Flash-Lite & Gemma 4 | 2026-07 / 2026-04 | Vendor + AA benchmarks | 3.5 Flash-Lite: ~350 tok/s, AA Index 12, ~45.5% SimpleQA, `thinking_level=minimal` for JSON; Gemma 4 26B-A4B: 82.3% GPQA, 88.3% AIME'26. | [Google Blog](https://blog.google/technology/ai/gemini-3-6-flash-3-5-flash-lite/), [DeepMind](https://deepmind.google/technologies/gemma/) |
| Mistral Small 4 (`2603`) & DeepSeek V4 (`0731`) | 2026-03 / 2026-07 | Vendor + OR + AA | Mistral Small 4: 0.38s p50, 136 tok/s; DeepSeek V4: AA Index ~53, 86.2% MMLU-Pro (Flash), 100–168 tok/s (`0731` Flash). | [Mistral Docs](https://docs.mistral.ai/getting-started/models/mistral_small_4/), [OpenRouter](https://openrouter.ai) |
| r/LocalLLaMA, r/LLMDevs & HN Practitioners | 2026 | Practitioner anecdotes | Prompt-only JSON fails >60% (fences/prose); reasoning models leak CoT into JSON without `json_schema`; putting verdict *before* reason in JSON degrades accuracy; voting/CoT stabilizes judges at latency cost. | [Reddit](https://reddit.com), [HN](https://news.ycombinator.com) |

## C. Recommendations per Job

### 1. JUDGE (closed-choice directness / faithfulness / correctness; <= ~1.5s runtime, any offline)
- **Primary:** `mistralai/mistral-small-2603` ($0.15/$0.60, 0.38s p50, 136 tok/s, supports `logprobs` + `json_schema` + low reasoning effort). Hits the <=1.5s runtime bar AND preserves `logprobs` for confidence calibration.
- **Backup:** `anthropic/claude-haiku-5.5` ($0.10/$0.50, `effort: "low"`). Strong reasoning/instruction adherence, but ~1.1–1.4s TTFT leaves tight margin for runtime <=1.5s and lacks standard `logprobs` (use 3-sample majority vote or rubric score offline).
- **Cheapest "good enough":** `google/gemma-4-26b-a4b-it` (~$0.08/$0.26, supports `logprobs` + `json_schema`, thinking off for runtime). Keep `qwen/qwen3.8-27b` on the lab server for zero-marginal-cost offline judging.
- **Failure modes to test:** (1) verbosity/style bias rewarding long fluent refusals/dumps, (2) verdict-before-reason JSON ordering locking in wrong label, (3) adaptive thinking tokens pushing runtime >1.5s or refusing single-letter output, (4) missing `logprobs` on chosen OR host returning HTTP 400.

### 2. SPLITTER (compound question -> doc question + web question + placeholder; strict JSON, <= ~2s)
- **Primary:** `google/gemini-3.5-flash-lite` ($0.30/$2.50, `thinking_level: "minimal"`, native `json_schema`, ~350 tok/s). Already proven in ME4 stack for fast structured calls; minimal thinking prevents reasoning latency spikes.
- **Backup:** `anthropic/claude-haiku-5.5` ($0.10/$0.50, `effort: "low"` + strict `json_schema`). 3x cheaper input than Gemini 3.5 Flash-Lite with strong decomposition logic; watch TTFT (~1.1–1.4s) against the 2s budget.
- **Cheapest "good enough":** `openai/gpt-5-nano` ($0.05/$0.40, `effort: "minimal"`, strict `json_schema`) or `mistralai/mistral-small-2603` ($0.15/$0.60, 0.38s p50).
- **Failure modes to test:** (1) JSON drift / markdown fences or leaked `<think>` tags if `json_schema` is dropped by an OR fallback host, (2) over-splitting simple single-source questions, (3) malformed placeholder binding across doc/web sub-queries, (4) `400` errors from passing `temperature` to Haiku 5.5.

### 3. SHORT REASONING (multi-step table math, comparisons, doc-vs-web conflicts; <= ~4s runtime, any offline)
- **Primary:** `anthropic/claude-haiku-5.5` ($0.10/$0.50, `effort: "medium"`). 89.6% GPQA Diamond reasoning at budget pricing fits comfortably inside <=4s runtime for short answers if effort is capped at `low`/`medium`.
- **Backup:** `deepseek/deepseek-v4-flash-0731` (~$0.045/$0.14–$0.30, 0.38–0.85s latency, 100–168 tok/s, `effort: "medium"`, `reasoning.exclude: true`). Extremely high reasoning-per-dollar, contingent on ZDR host filtering.
- **Cheapest "good enough":** `google/gemma-4-26b-a4b-it` (~$0.08/$0.26, thinking enabled) or `qwen/qwen3.8-flash` ($0.15/$0.47).
- **Failure modes to test:** (1) thinking-token runaway blowing past 4s runtime and inflating output cost 5–10x, (2) conflict resolution hallucinating parametric facts when docs and web disagree (remember Haiku 5.5's 23.8% SimpleQA), (3) `reasoning.max_tokens` unsupported on models using `effort` only.

## D. Minimal Bake-off Plan (< $0.50 Total)

- **Dataset (36 items):** 20 **Judge** items (10 known-bad Pi spoken turns from `20261126-173512.jsonl` + 10 golden-v2/dev rows across directness/faithfulness/correctness), 10 **Splitter** compound doc+web queries (incl. 3 non-compound controls), 6 **Short Reasoning** items (2 table math, 2 multi-doc comparison, 2 doc-vs-web conflicts).
- **4 Models to test:** `mistralai/mistral-small-2603`, `anthropic/claude-haiku-5.5`, `google/gemini-3.5-flash-lite`, `google/gemma-4-26b-a4b-it` (or swap `deepseek/deepseek-v4-flash-0731` if ZDR host confirmed).
- **Metrics:** Gold-label agreement (%), Cohen's kappa ($\kappa$) on Judge, valid-JSON & placeholder pass rate on Splitter, exact/rubric accuracy on Reasoning, p50 & p95 latency (s), $/1k calls, and run-to-run stability (3 repeats on the 20 Judge items, temp=0 where supported).
- **Budget (< $0.50):** 36 items + 40 judge repeats = 76 calls/model (304 calls total). At ~1,200 input + ~250 output tokens/call (~365K in, ~76K out total), worst-case cost across all 4 models is **~$0.09–$0.14** (even with 2x reasoning-token overhead + 5.5% OpenRouter fee: **< $0.25**).

```python
# OpenRouter bake-off call (read key from env; never hardcode or log secrets)
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {
    "model": "mistralai/mistral-small-2603",  # or anthropic/claude-haiku-5.5, google/gemini-3.5-flash-lite
    "messages": [{"role": "user", "content": prompt}],
    "reasoning": {"effort": "low", "exclude": True},  # omit/minimal for Splitter; low/medium for Reasoning
    "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    "provider": {"zdr": True, "require_parameters": True},  # set logprobs=True, top_logprobs=5 only where supported
}
# POST https://openrouter.ai/api/v1/chat/completions
```

## E. Risks

1. **ZDR / Upstream Data Leakage:** On **2026-11-22**, OpenRouter off-boarded 3 providers and stripped ZDR from **125 endpoints** after a Dragos canary-token audit (Jul–Nov 2026) exposed undisclosed reseller routing hops ([OpenRouter](https://openrouter.ai/docs/announcements/data-processing-incidents-on-inference-providers), [Eden AI](https://www.edenai.co/blog/openrouters-zdr-incident-what-failed-and-how-to-check), [TNW](https://thenextweb.com/news/openrouter-drops-three-ai-providers-after-dragos-catches-prompt-theft)). **Mitigation:** Always set `provider.zdr: true`, `allow_fallbacks` with explicit `only`/`order` vetted providers (e.g., Anthropic, Vertex, Azure), and never send sensitive docs to `:free` endpoints.
2. **Outages, Rate Limits & Mid-Stream Errors:** OpenRouter has no contractual SLA (35–50 min core incidents observed in 2026) and upstream 429s mid-stream arrive as SSE `finish_reason: "error"` after HTTP 200. **Mitigation:** Configure model-level fallbacks and strict runtime timeouts (`<= 1.5s` / `<= 2.0s` fail-open).
3. **Silent Deprecations & Parameter Drift:** OpenRouter drops discontinued preview/free slugs immediately from `/api/v1/models` without long grace periods; `require_parameters: true` prevents silent `json_schema`/`logprobs` drops but narrows provider pools. **Mitigation:** Pin GA slugs over `-preview`, poll `/api/v1/models` `supported_parameters` before bake-offs, and keep `Qwen3.8-27B` local as zero-network offline fallback.
