# ME4 Pi re-run (2026-11-25, commit 53cc38b): do exactly this, in this order

You are the agent on the Raspberry Pi. This is a RUN-AND-MEASURE job: change no repo file, make no commit, create no branch.
If a command fails: copy the exact command and error into the report and go on to the next step that does not depend on it.
Never print a key or token. Never open a file whose name contains `sealed`. No paid calls. `sudo` is not needed.
Write everything to `~/pi-check/REPORT2.md` AS YOU GO. Paste raw command output, not summaries.

## What changed since your last run (so you know what you are checking)
1. IPv4 is now forced inside the agent (`pipeline/src/voice_pipeline/net.py`): do NOT use your old IPv4 shim any more.
2. `--docs-first` and `--docs-db` are real command-line options now, and work with no router (`NO_ROUTER=1`).
3. `file_context` (the document lookup) was rewritten to be about 18x faster on the laptop (1.23 s on the Pi last time).
4. Zip ingest accepts big zips; OCR of PDF scan pages was fixed (PNG bytes); new indexes in the database.

## Step 0. Put the new code in place (the owner runs the sending side)
The owner sends the code from the laptop with: `git archive 53cc38b | ssh <pi> 'tar -x -C <the checkout folder you used last time>'`
(the same folder that already has `.venv` and `pipeline/.venv` synced, so nothing is reinstalled). Then check, inside that folder:
    grep -c "def prefer_ipv4" pipeline/src/voice_pipeline/net.py            # must print 1
    grep -c "def question_like" pipeline/src/voice_pipeline/doc_route.py     # must print 1
    grep -c "idx_chunk_rel_ord" src/voice_agent/rag/ingest.py                # must print >= 1
    ls scripts/me4_pi_turns.py scripts/me4_pi_check.py                       # both must exist
If any grep prints 0 or a file is missing: STOP and report "code not updated". Do not run `uv sync`; use `UV_NO_SYNC=1` on every `uv run` (it re-downloaded 1 GB last time).
Make sure `PYTHONPATH` does NOT include `~/pi-check/shim` (run `echo $PYTHONPATH`; unset it if it does).

## Step 1. Facts (1 min)
    uname -m; nproc; free -m; df -h / /tmp; vcgencmd get_throttled; python3 --version; tesseract --version | head -1
STOP only if free disk is under 1.2 GB.

## Step 2. IPv4 check (1 min) - the bug that made every turn say "I can't reach the internet"
    getent ahosts generativelanguage.googleapis.com | head -4
    UV_NO_SYNC=1 PYTHONPATH=pipeline/src:src .venv/bin/python - <<'PY'
    import socket, time
    def t(label):
        t0 = time.time()
        try:
            socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=15).close()
            print(label, "connect", round(time.time() - t0, 2), "s")
        except OSError as e:
            print(label, "FAILED", e, round(time.time() - t0, 2), "s")
    t("before prefer_ipv4")
    from voice_pipeline import net
    net.prefer_ipv4()
    t("after prefer_ipv4")
    PY
Expected: "before" about 8 s (broken IPv6), "after" under 0.5 s. Report both numbers. If `.venv/bin/python` cannot import `voice_pipeline`,
use `pipeline/.venv/bin/python` instead and say which one worked.

## Step 3. Ingest + integrity + OCR timing (about 3 min)
The 44-file zip is at `~/ipo44.zip` (if not, build it: `cd ~/rag-testsets/ipo-2026/raw && zip -qr ~/ipo44.zip . -x '.rag_cache/*'`).
    mkdir -p ~/pi-check/work3
    PYTHONPATH=src UV_NO_SYNC=1 .venv/bin/python scripts/me4_pi_check.py ~/ipo44.zip --work ~/pi-check/work3 --out ~/pi-check/pi-check3.json
Required: `ingest.last_line` says `44 files, 44 ok, ... 13894 chunks`; `integrity.index_matches_chunks` is true (if false: STOP, report); `ingest.wall_s`;
`ingest.parent_peak_mb`; `queries.median_s`; `ocr.median_s` and `ocr.images_with_text`.
Then: `sqlite3 ~/pi-check/work3/docs.db ".indexes chunk_data"` must list `idx_chunk_rel_ord`.
The extracted files are in `~/pi-check/work3/raw` (used next).

## Step 4. Document lookup speed (1 min)
    PYTHONPATH=src UV_NO_SYNC=1 .venv/bin/python - <<'PY'
    import time, statistics
    from pathlib import Path
    from voice_agent.rag.dbindex import DbIndex
    w = Path.home() / "pi-check/work3"
    db = DbIndex(w / "docs.db", raw_root=w / "raw")
    qs = ["How quickly do I need to notify GSA IT after a suspected incident?",
          "Where does my base schedule live?", "What was the US population in 2024?",
          "What is the GS-7 step 3 salary?", "Which file mentions a deadline?"] * 4
    ts = []
    for q in qs:
        t0 = time.perf_counter(); db.file_context(q); ts.append(time.perf_counter() - t0)
    print("file_context median", round(statistics.median(ts), 3), "max", round(max(ts), 3), "n", len(ts))
    PY
Last time: 1.23 s median. Report the new median.

## Step 5. Start the stack in the new mode (no router, docs-first, no extras)
    cd <the checkout folder>
    NO_ROUTER=1 UV_NO_SYNC=1 VOICE_AGENT_VERIFY_MAX=0 \
      AGENT_ARGS="--docs-first --hedge -1 --docs-db $HOME/pi-check/work3/docs.db --docs-dir $HOME/pi-check/work3/raw" \
      bash scripts/stack/up.sh text
Wait until `ss -ltn | grep 8781` shows the port (up to 60 s). Logs are in `out/stack/`. If it does not start, paste the last 30 lines of the agent log and STOP this step.
Record `free -m` right after it is up.

## Step 6. Ten timed turns (about 2 min)
    PYTHONPATH=src UV_NO_SYNC=1 .venv/bin/python scripts/me4_pi_turns.py --out ~/pi-check/turns2.json
It prints one JSON line per turn and a final `N/10 ok_hint` line. Paste ALL ten lines into the report. `ok_hint` is only a hint:
for each turn also write one sentence saying whether the answer is right, using the question's reason in the `why` field.
Any answer containing "can't reach the internet" means IPv4 is still broken: report it and run Step 2 again.
Then check which path ran: `grep -c "route" out/stack/*agent*.log` and paste the last 15 log lines that show `route=` (docs / routed / agent).

## Step 7. Stop and report
`bash scripts/stack/down.sh text` if it exists, otherwise `make text-down`. Then finish `~/pi-check/REPORT2.md` with this table (fill every cell, "n/a" only with a reason):

| measure | last run (47817d7) | this run |
|---|---|---|
| IPv6 connect to Gemini | 8.2 s | step 2 "before" |
| after prefer_ipv4 | n/a | step 2 "after" |
| ingest 44 files, wall | 77-87 s | |
| parent peak MB | 159-162 | |
| file_context median | 1.23 s | |
| tesseract median / 30 images | 0.19 s | |
| turns right (of 10) | 4/5 | |
| first answer text, doc turns | 1.7-2.6 s | |
| first answer text, weather | 0.9 s | |
| free memory with stack up | 2,556 MB | |

End with "Suggested changes" (do not make them) and the list of any step you could not do and why.
