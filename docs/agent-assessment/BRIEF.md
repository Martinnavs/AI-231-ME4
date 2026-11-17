# Brief: independent assessment of the voice/text agent, and how to improve it and prepare it for RAG

You are one of several independent reviewers. Work READ-ONLY: do not edit, create or delete any file, do not commit, do not install packages, do not start the agent stack, and make no network or paid API calls (no Gemini, Tavily, OpenRouter). Never open `~/.secrets/` or any key/token file. Running the free offline unit tests is allowed but optional (`.venv/bin/python -m pytest -q`, `cd pipeline && .venv/bin/python -m pytest -q`; one known failure needs a speech server).

Your final answer (printed as your last message) is the deliverable: one Markdown report. Be detailed and concrete. Every claim about the code cites `path:line`; every claim from a reading cites the file and its section or module (e.g. `context-engineering/source-notes/M1_intro-context-engineering.md §6`). Mark anything you inferred but did not verify as **unverified**. Do not pad: evidence over adjectives.

## The project

Repository: `/mnt/School/PhD/AI231/AI-231-ME4`, branch `text-only` (the voice agent driven from a terminal chat; the speech side is settled).
- Front agent server: `pipeline/src/voice_pipeline/` (start at `agent_server.py` `AgentServer.handle`, then `agent.py`, `agent_tools.py` `_recorded`, `call_policy.py`, `router.py`, `hedge.py`, `chain_state.py`, `doc_answer.py`, `scratchpad.py`, `payload_guard.py`, `spoken_filter.py`, `fast_path.py`, `venue.py`).
- Front agent prompt: `pipeline/config/eval/system_agent.txt` (+ `system_agent_case_f.txt`).
- MCP servers (tools): `src/voice_agent/mcp_servers/` (especially `tavily_search.py`, `gemini_grounding.py`, `docs.py`, `web_info.py`, `delegate.py`), the changeable-question gate `src/voice_agent/gate.py`.
- Text client: `src/text_chat/`. Document corpus: `corpus/baby-onit/`.
- Design docs to read: `docs/SEARCH-SPEC.md` (the current goal: search that never assumes, groundwork for RAG over files the user ingests whose freshness is unknown), `docs/CONTEXT-CRITIQUE.md`, `docs/TEXT-MODE.md`, `docs/MCP-SERVERS.md`, `docs/ARCHITECTURE.md` (skim), `.scratch/context-p1-p3/REVIEW.md`.
- Evaluation tooling: `scripts/search_eval.py`, `scripts/chain_eval.py`, `scripts/freshqa_gate_check.py`, `scripts/output_audit.py`, `scripts/capture_grounding.py`, `pipeline/src/voice_pipeline/dryrun.py`, `config/eval/`, `pipeline/config/eval/agent_prompts.jsonl`, `tests/`, `pipeline/tests/`.
- Runtime evidence (local logs, read only): `pipeline/out/agent_server/*.jsonl` (one row per turn), `out/text/logs/agent.log`, `~/.local/state/voice_agent/web_answer_ledger.jsonl` (grounded-search ledger), `~/.local/state/voice_agent/gate_outcomes.jsonl` (second-look verdicts).

## The theoretical readings (use them to justify recommendations)

- `/mnt/Obsidian/Claude/second-brain/resources/evaluating-and-improving-agentic-applications/` (read `NOTES.md` and `source-notes/M1..M5`)
- `/mnt/Obsidian/Claude/second-brain/resources/RAG/` (same layout)
- `/mnt/Obsidian/Claude/second-brain/resources/context-engineering/` (same layout)
- `/mnt/Obsidian/Claude/second-brain/resources/agentic-ai/` (same layout)
- `/mnt/Others/learning/extra-resources/`: notebook HTML exports. Plain-text extractions of all of them, and of the book *Hands-On RAG for Production* (`ragbook.txt`; use its table of contents near the top to find chapters 3, 6 and 7), are in `/tmp/claude-1000/-mnt-School-PhD-AI231-AI-231-ME4/b00e114e-0f9a-49aa-a4b8-37acc3feb7e2/scratchpad/res/`. Read those `.txt` files rather than the HTML/PDF.

## What to produce

1. **How the agent works now**: one turn end to end (fast path, router, hedged Gemini race, tools, post-turn work), what the model sees each turn (system prompt, tools, history, appended tails), memory within and across sessions.
2. **Inventory of every check and guard** on the way to a spoken answer (gates, lookups, retries, fallbacks, attribution, verification, document answering, output filters): what each guarantees, what it does not, where it is in the code.
3. **Problems and risks**, ranked by severity x frequency, each with code evidence and, where possible, evidence from the logs/ledgers. Classify each with the readings' taxonomies (the five agent failure modes; the four context failure modes; RAG retrieval/generation/ingestion failures).
4. **Readiness for RAG over user-ingested files**: status of each requirement in `docs/SEARCH-SPEC.md` (E1-E8, S1-S13, D1-D8): done / partial / missing, with evidence.
5. **Evaluation and observability**: what exists, what is missing, how it should be built (cite the readings).
6. **Recommendations, ordered**: for each, the reading(s) that justify it, the code it touches, the expected effect, the cost/latency trade-off, and a concrete acceptance test (a numeric threshold in an automated test where possible). Say what you would NOT do and why.
7. **Open questions** for the owner.
