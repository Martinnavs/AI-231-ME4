# Documentation index

> **This is the production tree.** Research notes, history, experiments (scripts, tests, recorded results) live on the development branch `agy/interrupts`; a file named in a doc below that is not here is there, or is a local file (`config/pi.env`, `config/profile.md`) or runtime output (`out/`).

Start with **`HANDOFF.md`**. Everything else is linked from there; this is the full list.

## The handoff set (read first)

| File | What it is |
|---|---|
| `DEEP-RESEARCH-HANDOFF-2026-11-26.md` | Research skill and isolated harness: branch, entry points, offline validation, limits, and remaining live integration |
| `GPT-SOL-CRITIQUE-2026-11-26.md` | Adversarial process and agent review at `8dbc4e3`, incorporating Pi REPORT4, with three linked Luna research reports on extraction, retrieval, and generation |
| `HANDOFF.md` | the map: state, findings, decisions, open questions, next actions, working agreements |
| `ARCHITECTURE.md` | the voice agent as deployed on 2026-11-05: topology and ports, lifecycle (`make pi-detect`, `up`, `talk`, `down`), one turn end to end, every component with its settings, state on disk, measured numbers against the industry, security, config table, differences from older docs |
| `PI-TEST-PLAN.md` | runbook for the Raspberry Pi echo test (what to run, how to read it, what each outcome means) |
| `PI-REMOTE-SPEAKER.md` | the laptop microphone with the Pi as loudspeaker: how the Pi speaker echo defeats Silero, the echo guard that fixes it, and how well it worked (6 of 6 interruptions, 0 of 6 false) |
| `(removed)` | the document search corpus (the document corpus, Explained): what is in it, how it is chunked and searched, where it is wired, what was checked |
| `LIVE-AGENT.md` | the agent (smart_home + docs MCP server) in the live voice loop: what was built, the stress dry run, the two spoken sessions (latency, tool routing, interruptions) and what broke (echo in the transcripts) |
| `AEC-REMOTE-SPEAKER.md` | echo cancellation across two devices (laptop mic, Pi speaker): the 137 ppm clock drift, offline results on real recordings, what is not shown |
| `SMART-HOME-TOOL.md` | the one `smart_home` tool for the 19 ME2 commands, the separate Pipecat project `pipeline/`, and the LLM choice: native Gemini 3.1 flash-lite + router prompt 50/50, first reply 0.52 s (plain) / 1.31 s (tool) |
| `MCP-SERVERS.md` | the MCP servers; the last section is the live agent's real tools (time, weather + Wikipedia, the Pi's device server), their security and how to start them |
| `PUBLISH-PAGES.md` | publishing a finished app to GitHub Pages after a spoken yes: owner setup, what is checked, what is not covered |
| `PROFILE.md` | the owner profile (`config/profile.md`): what it is, setup, how it is installed, the optional doctor check, the Pi, the live check |
| `LATENCY.md` | where an answer's time goes, the LLM spike probe, the research on fillers, and the safeguards (fast path, the tools' `say` sentence, doc_chat fillers with a give-up at 8 s): landed 2026-10-24/25 (the spoken sessions, R4, R5, R6 and what each changed are in its later sections) |
| `ONIT-PARITY.md` | the professor's added criterion (the voice agent must do all that OnIt can do): OnIt and the document corpus inventoried, the minimum MCP servers and tools to replicate the fundamentals, voice-specific issues, open questions (analysis only, 2026-10-27) |
| `ROUTER-WEB-AND-DOCS-RACE.md` | the router's web lookups (cluster tickets 01/02, kd-web2 candidate), the documents-vs-web race in the harness, measurements and a latency estimate (2026-11-03) |
| `AGENT-ASSESSMENT.md` | **master document for improving the agent and preparing it for RAG** (2026-11-18): one turn end to end, every check with its code location, measured evidence, ranked problems, SEARCH-SPEC readiness, the readings mapped to the code, roadmap with acceptance tests |
| `agent-assessment/` | the two independent assessments behind it (codex gpt-6.1-terra, agy gemini-4-argon-high), verbatim, the shared brief, and `VALIDATION.md` (every claim checked; read it before citing either report) |
| `TURN-TRACE.md` | the per-turn trace row of the agent server (`out/agent_server/<run>.jsonl`): turn and session ids, call spans, tail, tokens, ledger slice, the second-look file `checks/<run>.jsonl`; the contract the evaluators and the benchmark read |
| `AGENT-BENCHMARK.md` | the behavioural benchmark of the agent (39 cases, code evaluators, offline run and guarded live run): case schema, commands, report, known failures |
| `RAG.md` | chat with your own documents (2026-11-18, updated 2026-11-19): the drop folder `~/agent_sandbox/docs`, how to ask and what answers look like, the docs route and its checks, canonical Markdown pipeline and formats (direct readers for pptx, xlsx, html, epub; optional MarkItDown arm), DocIndex, FTS5 / BM25 and the hybrid retriever (rank-bm25 arm) with its thin embedding env, settings, evaluation (golden sets, metrics CLI, converter-arm measurements), security notes, known limits |
| `RAG-PI.md` | RAG on the Raspberry Pi (2026-11-19): the `pi-rag` group (rank-bm25, pypdf, openpyxl, python-pptx, lxml, defusedxml), all nine formats via direct readers, `scripts/pi_rag_check.py` and how to read it |
| `RAG-PI-HANDOFF.md` | Raspberry Pi test handoff (2026-11-20): hypotheses, install, caveats, a 10-step test plan with exact commands, x86 reference numbers and a report format |
| `RAG-EXPERIMENTS.md` | RAG experiment log E1-E12 (2026-11-18..20): protocol, every measurement with numbers and conclusions, open hypotheses; reference per-query results in `rag-reference/` |
| `ME4-SPEC.md` | The ME4 specification as given (verbatim), requirement-by-requirement status (R1-R10), critic-driven latency work and the ranked gaps for the day of checking |
| `ME4-NEXT-PLAN.md` | Work-package plan for ME4 (WP0-WP11): zip ingestion, cold start, documents-first multi-quote answers, act-then-critique (OnIt-style) validation, tables, tokenizer, formats; owner decisions and the day-of dry run |
| `PROGRESS-2026-11-11.md` | **latest:** the harness review and fixes, owner profile file, Cebu venue and barangay, nearby/nearest places (Photon, Google), endpoint silence 0.7, GitHub Pages publish (ticket 04); evidence, how to run, open items in order |
| `../finetune/README.md` | start here on the cluster: the router fine-tuning project (contract, evaluation sets, baselines, tasks, rules); code in `finetune/` |

## Speech and the test client

| File | What it is |
|---|---|
| `LAPTOP-SPEECH.md` | the laptop speech server: benchmark results (Kokoro, Supertonic, Kroko), the WebSocket protocol |
| `STT-HOSTING.md` | STT accuracy and hosting findings: Parakeet, hotwords, name repair, padding, live sessions, latency, early decode |
| `DOC-CHAT.md` | the voice test client: every option, the improved loop, barge-in, Gemini, ask-to-add |
| `IRL-RECIPE.md` | a real-voice test with recording and a report |

## Turn end and interruptions

| File | What it is |
|---|---|
| `BARGE-IN.md` | the original barge-in design and the measurements on the laptop (echo with and without cancellation) |
| `INTERRUPTS.md` | the model-free interruption bench, the detector ablation, what it shows |

## Project 2 design (from `main`)

| File | What it is |
|---|---|
| `../.scratch/project2-voice-agent/DESIGN.md` | the Pi voice agent design, decisions, the professor's requirements |
| `../.scratch/project2-voice-agent/tickets/` | tickets 01-28 (12 = barge-in gate; 21-24 = laptop speech track; 25-28 added 2026-10-23; order after the reassessment at the end of `DESIGN.md`) |
| `../.scratch/live-agent/`, `../.scratch/echo-fix/` | plans, level-3 tickets, the agy briefs and replies, the reviews, edit lists and ticket generators for the agent and for the echo canceller and duck |
| `../.scratch/smart-home-tool/` | plan, design, level-3 tickets, agy fence and results for the `smart_home` tool |
| `../PROJECT2-DESIGN-REVIEW.md`, `../FULL-DUPLEX-RESEARCH.md` | the reviews behind the design |
| `AUDIO-ENGINE.md`, `EVENT-LOG.md`, `MCP-SERVERS.md`, `BENCHMARK.md` | contracts of the main project |

## Research notes from the agy assistant (`.scratch/speech-hosting/`)

`AGY-RESEARCH.md` (hosting), `AGY-STT-IMPROVE.md`, `AGY-PARAKEET-IMPROVE.md`, `AGY-PARETO-REPAIR.md` (STT), `AGY-NET-*.md` (poor-network practice), `AGY-SPEC-DECODE.md` (early-decode accuracy), `AGY-ECHO.md` (echo), `AGY-GEMINI-FAST.md` (Gemini models for a fast agent), `AGY-CHEAP-LLM.md` (cheap hosted LLMs from the Philippines; relies on 2024-era data), `AGY-PI-NEXT-v1.md` / `AGY-PI-NEXT.md` (one-word interrupt and AEC, **unreliable**: see `RESEARCH-HANDOFF-PI.md`), `PROMPT-PI-NEXT.md` (the brief), `VERIFICATION.md`. They are web-search summaries; claims are tagged by the assistant and partly unreliable (see `ECHO-RESEARCH.md` "How far to trust it").

## Results coming back from the Pi

`docs/pi-results/` (created by `scripts/pi_pack_results.sh`): one folder per run with `run.md` and `run.json`, plus your `NOTES.md`.

## Not in the repository (the owner's, local)

`docs/VOICE-AGENT-EVALUATION-METRICS.md`, `kek.txt`, `resources/building-voice-agents/` are the owner's untracked files; leave them alone.
