# Research skill and harness handoff (2026-11-26)

Implemented as an opt-in library on branch `feat/deep-research-skill-harness`, in worktree `/tmp/ai231-deep-research-skill`, based on `c80522c`. The owner's active checkout and ordinary voice route are unchanged. This branch includes the updated GPT-SOL critique and supporting research notes so a reviewing agent has both rationale and code. It is a working offline-tested foundation, not a deployed research service or a measured quality/latency improvement.

## What is available?

| File | Role |
|---|---|
| `skills/deep-research/SKILL.md` | Research procedure, loaded explicitly for research jobs; not added to every voice turn |
| `src/voice_agent/research/runner.py` | Bounded plan/search/synthesize loop with one optional evidence-gap round, source deduplication, citation-ID checks, trusted request time, skill hash/configuration provenance, phase and snapshots |
| `src/voice_agent/research/adapters.py` | Adapter to the actual `ToolRegistry`/`web_research` result format; no new search provider |
| `src/voice_agent/research/jobs.py` | Dedicated background job store reusing `JobTable`, explicit status/result/cancel, saved reports/evidence, bounded active jobs |
| `pipeline/src/voice_pipeline/research_adapter.py` | Caller-owned Gemini client/model adapter with JSON responses and capped output tokens; separate from document-answer schema |
| `scripts/research_demo.py` | Fully offline synthetic example, with no keys, model inference, or network |
| `tests/research/` | Offline limits, evidence, deadline, cancellation, lifecycle and transport tests |

No external dependency was added. The runner is standard-library code; the adapter uses the existing MCP client and optional pipeline Gemini installation. This repo previously had no runtime skill loader. The explicit `Runner(skill_path, ...)` read is this feature's initial loader; it is not a generic plugin registry or automatic skill discovery system.

## How does it behave?

The default profile has a single 30-second monotonic deadline including planning and search, at most 4 search-adapter calls, at most 3 model calls, concurrency 2, 8 retained sources, 2,000 text characters per source, and 12,000 retained evidence-text characters (metadata is separately bounded per source). Query/source duplicates are suppressed. At most two distinct queries are selected per round and one follow-up round can run. Limits are configurable and are experimental bounds, not latency promises.

The model first proposes queries. The runner searches and assigns source IDs, then requests claims tied to those IDs and explicit unresolved gaps. A gap can trigger one follow-up search and final synthesis if budget remains. Claims with missing or nonexistent evidence IDs are omitted. No evidence produces a partial result. Malformed/oversized model output fails explicitly. Timeout and cancellation persist the latest snapshot.

Evidence includes URL/title/exact retained text, publication date when returned, retrieval time, and evidence kind. Current MCP results are **search snippets**, not full-page reads; the report labels that distinction. Claim-to-source semantic support, truth, temporal interpretation, and exhaustive coverage are not proven by ID validation. The skill asks for temporal care, while the larger claim-validity storage proposal remains research/design work.

The runner never accepts a model-selected tool name: it only invokes the injected model and search callables. The shipped adapter verifies the selected registry entry is `web_research`. It does not expose shell, builder, home-command or publishing tools. Research writes only its configured job artifacts. The code makes no arbitrary URL fetches; a future source reader needs its own bounded network policy.

## Run the offline example

From the isolated worktree, use the existing root environment or an equivalent installed environment:

```bash
PYTHONPATH=src /mnt/School/PhD/AI231/AI-231-ME4/.venv/bin/python \
  scripts/research_demo.py --out /tmp/research-skill-demo
```

The demo creates a dedicated job directory containing `report.md` and `evidence.json`, plus JobTable state and a completion notice. Its example.org source and result are synthetic. Reusing the same demo state directory creates another numbered job. Do not point it at the existing delegate state directory.

## How should the main agent integrate it?

The integration is deliberately explicit and does not require a new framework:

```python
from pathlib import Path
from voice_agent.research.adapters import McpSearch
from voice_agent.research.jobs import ResearchJobs
from voice_agent.research.runner import Runner
from voice_pipeline.research_adapter import gemini_research_request

# Existing caller-owned Gemini client and ToolRegistry, in their live event loop.
request = gemini_research_request(client, model_id, max_output_tokens=1500)
search = McpSearch(tools, tool="web__web_research", recent=False)
runner = Runner(Path(checkout) / "skills/deep-research/SKILL.md", request, search)
jobs = ResearchJobs(Path(research_state_dir), runner)
job = jobs.start(question, timezone="Asia/Manila")
# Expose jobs.status(job.id), jobs.result(job.id), await jobs.cancel(job.id).
# Call await jobs.close() before closing the model/MCP clients on shutdown.
```

`client`, `model_id`, `tools`, `checkout`, `research_state_dir`, and `question` are supplied by the application. The MCP prefix depends on deployment configuration; select the registry's actual web_research entry. Do not pass `ask_documents.gemini_json_request`: it fixes a document-answer schema incompatible with research plans/claims. The new Gemini adapter reuses the current client/authentication without modifying the existing specialist.

Next integration work for the owning agent:

1. Wrap model/search callbacks in existing cost/credit admission policy. The runner enforces adapter-call limits, **not** token input totals, provider-internal search fan-out, or a dollar ceiling. Set a provider-aware token/cost reservation before enabling paid jobs. Gemini output tokens alone are not a total-price bound, especially for thinking models.
2. Expose start/status/result/cancel through the chosen MCP or application tool registration. The library has methods, not a registered server. Keep research outside `AgentServer.lock` while it works and keep underlying clients alive for the job lifetime. Current adapter I/O is asynchronous, but persistence is small synchronous local-file I/O; move disk-heavy processing to a worker service if measurements require it.
3. Add one explicit research route and its short prompt hint, preserving one-fact/command paths. Supply source scope and resolved temporal interval when extending beyond the current public-web-only adapter. The runner's timestamp is captured per job; `ResearchJobs.start` currently uses host time and accepts timezone, while direct `Runner.run` also supports an injected aware reference time.
4. Bridge completion notices and result paging into the voice UI, accounting for its speech filter and report size. Task cancellation is distinct from interrupting speech. This library does not activate a new voice command or modify the existing chain state machine.
5. Test the actual router-on audio path, concurrent command latency, report support/freshness, and paid cost on owner-approved data before enabling broadly.

## What does cancellation and persistence guarantee?

A dedicated `ResearchJobs` instance owns a separate state directory in one process. The active-job cap rejects excess jobs rather than silently queueing them. `cancel` cancels its asyncio task and waits for cleanup; snapshots show cancellation if execution began. Immediate cancellation before the task starts still marks the job cancelled. Completed jobs remain addressable by exact ID. Immediate cancellation before the worker starts leaves only the terminal job record, with no evidence/report file.

The current MCP search server can shield shared work and retry internally. Cancelling this client cannot guarantee the remote request stops or refund work already accepted. Deadlines assume adapters cooperate with asyncio cancellation; this harness is not a process-kill or remote-billing guarantee. No new runner calls should be scheduled after cancellation. A future distributed service needs its own stronger lifecycle policy.

JobTable marks interrupted running jobs lost on restart. Evidence snapshots are preserved, but automatic resume is intentionally absent: repeating an uncertain paid request could duplicate cost. JobTable's compact status uses existing `done/error/cancelled/lost`; `evidence.json` carries the more specific runner state (`partial`, `timed_out`, `failed`, etc.). Read that field when presenting incomplete research. No cross-process concurrent writer support is claimed. A default 100-record cap rejects new submissions once reached; archive the dedicated store deliberately before starting a fresh one. No automatic deletion is performed.

## Validation and research

Final offline validation:

- Root research suite: **22 passed** using the existing root `.venv`.
- Gemini adapter suite: **2 passed** using the existing pipeline `.venv` (the root environment lacks Pipecat required by pipeline test fixtures).
- Ruff lint and format checks pass for all new Python files.
- Strict mypy passes for the four research modules and pipeline adapter (five source files).
- The offline synthetic demo completes and writes a report and evidence ledger.
- Existing configuration tests also passed (six tests); no broad unrelated suite was run.

No paid API tests or Pi/audio benchmarks are part of this branch. The source-referenced procedural skill is loaded in the test/demo path; it has not been shown to improve a real model's research answers yet.

Supporting decisions:

- [GPT-SOL critique](GPT-SOL-CRITIQUE-2026-11-26.md): prompt repair, dynamic temporal contracts, skill/harness ownership.
- [Framework comparison](DEEP-RESEARCH-INTEGRATION-2026-11-26.md): reuse existing tools first; packages are optional reference/comparison candidates.
- [Provider comparison](DEEP-RESEARCH-PROVIDERS-2026-11-26.md): Gemini Deep Research versus composable search/extract providers.
- [Prompt/temporal research](PROMPT-TEMPORAL-RESEARCH-2026-11-26.md): evidence-driven correction and date semantics.
