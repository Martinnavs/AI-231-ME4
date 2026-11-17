# Independent assessments of the agent (2026-11-18)

Two other models assessed the front agent and its readiness for RAG, independently and read-only, from one brief. Their full reports are kept here with a claim-by-claim validation, so the exercise does not have to be run again. The master document built from all three views is `../AGENT-ASSESSMENT.md`.

| File | What it is |
|---|---|
| `BRIEF.md` | the brief both models got, verbatim (scope, rules, sources, required sections) |
| `CODEX-TERRA.md` | codex `gpt-6.1-terra`, reasoning xhigh, sandbox read-only: 06:48-06:53, 210,463 tokens, 56 KB. Verbatim, with a one-paragraph banner |
| `AGY-ARGON.md` | agy `gemini-4-argon-high`, plan mode: 06:48-07:09, 150 KB. Verbatim, with a one-paragraph banner |
| `VALIDATION.md` | every claim checked against the code, logs and ledgers: codex 17/17 confirmed; agy 17 confirmed, 4 partly, 12 wrong; three claims reproduced by running the code |

**How to read them.** Start with `VALIDATION.md` §1. Cite codex after a spot check. Use agy as a source of leads: its stale-chain finding (F13) is the most serious new problem the exercise found, but its report also invents log lines, ledger fields and code text.

## Raw transcripts (not in git)

`.scratch/agent-assessment/transcripts/` holds the full working transcripts, kept locally and git-ignored (`.gitignore`) because they include excerpts of the copyrighted RAG book and the contents of local logs (the owner's test questions):

| File | Source |
|---|---|
| `codex-terra.log` (0.9 MB) | the codex session's stderr: every command it ran and every file excerpt it read, then the final report |
| `agy-argon-steps.txt` (3.6 MB) | best-effort text of agy conversation `f51e605d-ee82-4acd-9507-95b280b1b4a4` (134 steps), extracted from the protobuf blobs in `~/.gemini/antigravity-cli/conversations/f51e605d-....db` |

The readings were given to both models as plain-text extractions in a session scratch folder (`/tmp/claude-1000/.../scratchpad/res/`, gone after the session). To re-create them: strip the HTML tags from `/mnt/Others/learning/extra-resources/*.html`, and `pdftotext -layout` the book.

## Re-running

`.scratch/agent-assessment/run.sh` launches both in parallel (outputs to the same folder; `timeout 5400`):

```bash
agy --mode plan --model gemini-4-argon-high --print-timeout 85m --add-dir <readings dirs> -p "$(cat BRIEF.md)" > AGY-ARGON.md
codex exec -m gpt-6.1-terra -c model_reasoning_effort=xhigh -s read-only --ephemeral --add-dir <readings dirs> -o CODEX-TERRA.md "$(cat BRIEF.md)"
```

Re-run only after material changes to the agent. Then validate the new reports the same way (`VALIDATION.md` §4 has the commands for the reproducible checks).

Privacy note: both models ran on their vendors' services and read local logs that contain the owner's test queries. Neither was allowed to open `~/.secrets/`; the codex transcript was scanned for key patterns (none found).
