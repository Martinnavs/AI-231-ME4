# ME4: what the agent must do on the Pi (explicit task list)

## What this task is, and is not
- It is a RUN-AND-MEASURE task. You implement nothing, edit no file in the repo, create no branch and make no commit.
- "Running it there" means exactly this: get the existing branch `qwen/ingest` onto the Pi as it is, run the commands below, and report what they print.
- If a command fails, do not fix the code. Report the exact error and the command, then continue with the next task if it does not depend on it.
- If you think a change is needed, write it under "Suggested changes" in the report; do not make it.

## Step 0. Get the code (exactly)
Accepted alternative when git access from the Pi is awkward: the owner's laptop sends `git archive qwen/ingest` over SSH; unpack it, run `git log`-less, and report the commit hash the archive was made from (it must be `7ddcd82` or later). It is read-only either way.

If the repo is not on the Pi yet, clone it with the GitHub URL the owner gives you (the repo is `Martinnavs/AI-231-ME4`).
Then, inside the repo:
    git fetch origin
    git checkout qwen/ingest
    git pull --ff-only origin qwen/ingest
    git log --oneline -1          # must show aa44e09 or a later commit; report the line
If `git pull` says the branch diverged or you have local changes: STOP and report `git status`; do not reset or stash anything.

You are an agent running on the Raspberry Pi 4 that will host ME4. Do these tasks in order, record real command output, and
stop at any STOP line. Do not guess numbers; every figure you report must come from a command you ran.

## Rules
- Repo: the checkout from Step 0, branch `qwen/ingest`, never modified.
- Never open or read any file whose name contains `sealed`. Make no paid or live Gemini call unless task 6 says so. Never print a key or token.
- No `sudo` except the single `apt-get install` in task 2. Do not stop or restart any service; do not change the CPU governor.
- Write results to `~/pi-check/REPORT.md` as you go (a timeout must not lose work). Paste raw numbers, not summaries.

## Task 1. Machine facts (2 min)
Run: `uname -m; nproc; free -m; swapon --show; df -h /tmp ~; vcgencmd get_throttled; python3 --version`.
Record them. STOP and report if `MemAvailable` is under 1,500 MB or free disk under 1.5 GB (say which). Do NOT delete anything to make room beyond `uv cache clean` and `pip cache purge`; never touch `~/.local/share/claude`.

## Task 2. Install (10 min)
1. `uv sync --group pi-rag` (or the project's Pi sync). Report any package that fails to install, with the error line.
2. If `sudo -n true` fails (no sudo), SKIP this step and write "tesseract: no sudo" (the OCR number then reads "no engine"). Otherwise `sudo apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng`; then `tesseract --version`.
3. Check RapidOCR: `uv run python -c "import rapidocr"`. If it fails, write "rapidocr: not installed" and continue (do NOT hunt for model files).

## Task 3. The measurement (10 min)
Get the 44-file test zip: `~/ipo44.zip` if the owner placed it; otherwise build it on the Pi from the same files with
`cd ~/rag-testsets/ipo-2026/raw && zip -qr ~/ipo44.zip . -x '.rag_cache/*'` (never include `.rag_cache`; the ingest must report `44 files`, if it says 90 the cache was included, rebuild). Then:
`PYTHONPATH=src uv run python scripts/me4_pi_check.py ~/ipo44.zip --work ~/pi-check/work --out ~/pi-check/pi-check.json`
Copy the printed lines into REPORT.md. Required readings: `ingest.wall_s`, `ingest.parent_peak_mb`, `integrity.index_matches_chunks`
(MUST be true; if false, STOP and report: it means the search index lost rows), `integrity.files`, `integrity.images`, `queries.median_s`, `ocr.*`.
Expected on a laptop: 15-18 s, 100 MB, 13,894 chunks. Do not "fix" differences; report them.

## Task 4. Resume check (2 min)
Run the same ingest command again on the same work dir (a re-run). Report wall time and that chunk count and search-index rows are unchanged.

## Task 5. Five text turns, timed (10 min)
`make text-up AGENT_ARGS=--docs-first` (needs the router GGUF and a Gemini key; if either is missing write "text stack not startable: <reason>" and skip to task 7).
Using `uv run python -m text_chat --url ws://127.0.0.1:8781`, point the docs folder at the unzipped test files and ask, noting seconds to the first answer sentence:
1. a question whose answer is in one document (pick a fact from a `.md` or `.docx` file you read yourself);
2. a question about a number in a spreadsheet or CSV;
3. "What is the weather today?" (must NOT answer from documents);
4. a question that is NOT in the documents (it must say the documents do not say, not invent an answer);
5. "Which file did that come from?" right after question 1.
Record each question, the answer, the seconds, and whether the answer was right according to the file you checked. `make text-down` afterwards.

## Task 6. Not authorised unless the owner says so
Any live paid call, OpenRouter, Tavily, or Gemini vision. Skip.

## Task 7. Report (5 min)
Write REPORT.md with: the facts, install problems, the JSON from task 3, task 4, the five turns, and a final list "numbers that differ
from the laptop by more than 3x". Commit nothing. Send the owner `~/pi-check/REPORT.md` and `~/pi-check/pi-check.json`.
