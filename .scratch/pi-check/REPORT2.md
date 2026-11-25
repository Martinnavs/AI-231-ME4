# ME4 Pi re-run report (REPORT2)
Code: archive of qwen/ingest dfc068f unpacked over ~/me4-ingest (no git on Pi). Step 0 greps: 1,1,1; both scripts present; PYTHONPATH empty.
## Step 1
aarch64, 4 cores, MemAvailable 3276 MB, free disk 2.5 GB, throttled=0x0, Python 3.13.5 (uv venv 3.12), tesseract 5.5.0
## Step 2
before prefer_ipv4 connect 2.15 s; after prefer_ipv4 connect 0.05 s (this run; .venv python imported voice_pipeline OK). getent ahosts lists IPv6 first.
## Step 3
{"exit": 0, "wall_s": 116.4, "parent_peak_mb": 159, "last_line": "ingest: 44 files, 44 ok, 0 failed, 0 skipped (disk/time), 13894 chunks, 159 MB RSS, 0.0s", "stderr_tail": ""}
{"files": {"ok": 44}, "chunks": 13894, "search_index_rows": 13894, "index_matches_chunks": true, "images": {"pending": 133}}
wrote /home/rpi-anavarez/pi-check/pi-check3.json
queries.median_s 0.04; ocr: {'engine': 'available', 'images': 30, 'total_s': 41.3, 'median_s': 1.04, 'p90_s': 2.81, 'images_with_text': 18, 'peak_rss_mb': 81}
.indexes chunk_data (sqlite3 CLI absent, used pragma index_list): idx_chunk_rel_ord present.
Step 4: file_context median 0.042 max 0.153 n 20
## Step 5
Started with: NO_ROUTER=1 UV_NO_SYNC=1 VOICE_AGENT_VERIFY_MAX=0 AGENT_ARGS="--docs-first --hedge -1 --docs-db ~/pi-check/work3/docs.db --docs-dir ~/pi-check/work3/raw" bash scripts/stack/up.sh text -> worked first time (up.sh run with bash because the archive lacks the exec bit). Port 8781 listening. free -m right after: used 1096, available 2700 MB.
## Step 6 (raw: ~/pi-check/turns2.json; 8/10 ok_hint)
{"q": "How quickly do I need to notify GSA IT after spotting a suspected security incident?", "first_s": 3.78, "total_s": 9.47, "answer": "Searching for GSA IT security incident notification procedures.  [tool:web_answer] As per  GSA IT policy, you need to notify them within one hour of spotting a suspected security incident  and provide all available information to assist the response team with triage. If email is unavailable, you should contact them through another method.", "ok_hint": true}
{"q": "Where does my base schedule live?", "first_s": 2.65, "total_s": 2.65, "answer": "As per work_schedules.md: Your base schedule lives in HRLinks.", "ok_hint": true}
{"q": "What was the estimated United States population in 2024 in the census file?", "first_s": 1.55, "total_s": 1.56, "answer": "As per nst_est2024_pop.xlsx: 340,110,988", "ok_hint": true}
{"q": "What's the GS-7 step 3 salary?", "first_s": 1.65, "total_s": 1.65, "answer": "As per gs_salary_table_2026.pdf: $45,980", "ok_hint": true}
{"q": "Which file did that come from?", "first_s": 2.82, "total_s": 2.82, "answer": "As per gs_salary_table_2026.pdf.", "ok_hint": true}
{"q": "What is the weather today?", "first_s": 3.93, "total_s": 3.96, "answer": "Checking the weather.  [tool:weather] Today in Cebu City: drizzle, 25 to 31 degrees Celsius, 99 percent chance of rain. ", "ok_hint": true}
{"q": "What is the GSA TTS policy on bringing pets to the office?", "first_s": 2.94, "total_s": 6.86, "answer": "Checking the GSA Technology Transformation Services policy on pets.  [tool:web_answer] Federal regulations generally prohibit bringing animals onto federal property, which includes GSA-controlled facilities. Exceptions are made for service animals assisting individuals with disabilities. ", "ok_hint": false}
{"q": "What was the average home electricity price in Guam?", "first_s": 4.53, "total_s": 6.91, "answer": "Checking the average home electricity price in Guam.  [tool:web_answer] As per Callmepower (the only source I found), As of November 2026, the standard residential electricity rate in Guam is approximately 42.6 cents per kilowatt-hour. For a typical household using 1,000 kilowatt-hours per month, this results in an estimated monthly bill of about $426. ", "ok_hint": false}
{"q": "Set a timer for five minutes.", "first_s": 0.01, "total_s": 0.01, "answer": "Timer set for 5 minutes.", "ok_hint": true}
{"q": "How many hours does a full-time TTS employee work in a two-week pay period?", "first_s": 1.6, "total_s": 1.61, "answer": "As per work_schedules.md: 80 hours", "ok_hint": true}
My judgement per turn (checked against the files on the Pi):
1 RIGHT value (1 hour; security_incidents.md says "within 1 hour") but taken from web_answer, not the documents; route="" in the turn log (slow: 3.78 s first, 9.47 s total).
2 RIGHT: HRLinks (the miss from the last run is fixed). 3 RIGHT: 340,110,988. 4 RIGHT: pdftotext of gs_salary_table_2026.pdf row 7 step 3 = 45,980.
5 RIGHT: gs_salary_table_2026.pdf (the file of Q4). 6 RIGHT: weather tool, no documents. 
7 WRONG per the question: pets not in the files, the agent called web_answer and gave a general federal-property answer instead of "the documents do not say".
8 WRONG per the question: Guam not in the table, agent answered from the web (Callmepower, 42.6 c/kWh) instead of refusing.
9 RIGHT: timer set (route=fast). 10 RIGHT: 80 hours.
Right: 8/10. No "can\x27t reach the internet" answers.
Route per turn (from pipeline/out/agent_server jsonl, field route): docs for turns 2,3,4,10 (docs=accepted); fast for 9; empty (agent path) for 1,5,6,7,8. grep "route=" in agent.log matched no lines in this build (the field is in the jsonl, not the text log).
## Step 7
Stack stopped with scripts/stack/down.sh text; no agent_server process left. Disk 2.4 GB free.
| measure | last run (47817d7) | this run (dfc068f) |
|---|---|---|
| IPv6 connect to Gemini | 8.2 s | 2.15 s (before prefer_ipv4) |
| after prefer_ipv4 | n/a | 0.05 s |
| ingest 44 files, wall | 77-87 s | 116.4 s |
| parent peak MB | 159-162 | 159 |
| file_context median | 1.23 s | 0.042 s (max 0.153) |
| tesseract median / 30 images | 0.19 s (7 with text) | 1.04 s (p90 2.81, total 41.3 s, 18 with text) |
| turns right (of 10) | 4/5 | 8/10 |
| first answer text, doc turns | 1.7-2.6 s | 1.55-2.65 s on docs-route turns (3.78 s on turn 1, web path) |
| first answer text, weather | 0.9 s | 3.93 s |
| free memory with stack up | 2,556 MB | 2,700 MB |
## Suggested changes (not made)
- Docs-first lets the agent fall through to web_answer for questions the documents do not answer (turns 7, 8) and even for a document fact (turn 1); the refusal path did not trigger.
- Weather first-answer went 0.9 -> 3.9 s; compare with the router-less hedge -1 setting.
- Ingest 116 s vs 77-87 s: more OCR text indexed (18/30 images with text now) and the new index probably cost time; not separated.
- IPv6 connect is 2.15 s now vs 8.2 s before without any change on the Pi (variable); prefer_ipv4 removes it.
## Not done
Nothing skipped. sqlite3 CLI is not installed, so .indexes was checked with pragma index_list.
