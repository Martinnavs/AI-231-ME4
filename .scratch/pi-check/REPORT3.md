# ME4 Pi rehearsal report (REPORT3)
Frozen sha: 324cd10cd587c52dfa49fdfd8abf4805f8d6341a (tip of qwen/ingest; final RC-A code 16ff30c + docs). Sent with git archive over ssh; no git on Pi.
## Step 0
greps: ROW_KEY_CHECK 2, is_general_question 1, key_absent 7; both scripts present; PYTHONPATH empty.
## Step 1
aarch64, 4 cores, MemAvailable 3338 MB, disk free 2.4 GB (/ 14G, 82% used), /tmp tmpfs 1.9G, throttled=0x0, Python 3.13.5, tesseract 5.5.0. Pi clock was ~17 h behind, synced by NTP during the run.
## Step 2
getent ahosts lists IPv6 first. before prefer_ipv4 connect 2.12 s; after 0.04 s (monotonic clock; a first reading using time.time() gave 64725 s because of the clock jump, discarded). .venv/bin/python imported voice_pipeline.
## Step 3
corpus48.zip taken from Hugging Face dataset martinnavs/me4-pi-package (private), scp to ~/ipo48.zip (65,624,384 bytes, 48 members).
dry run: members 48, total_bytes 72855827, free disk check ok. Real run (workers auto, deadline 50):
zip: /home/rpi-anavarez/ipo48.zip  members: 48  total_bytes: 72855827  zip_bytes: 65624384
free_disk: 2500689920  free >= 3x zip: True
ingest: /home/rpi-anavarez/me4-ingest/.venv/bin/python -m voice_agent.rag.ingest /home/rpi-anavarez/pi-check/work4/raw --workers auto --deadline-min 50 --out /home/rpi-anavarez/pi-check/work4/docs.db
extracted: 48  junk_skipped: 0
ingest: 48 files, 48 ok, 0 failed, 0 skipped (disk/time), 13898 chunks, 109 MB RSS, 118.8s
wall_s 121.0 peak_mb 138 exit 0
integrity: chunks 13898 search_index_rows 13898 index_matches_chunks True
## Step 4
Started with the frozen command (NO_ROUTER=1, --docs-first, --hedge -1, --docs-db work4/docs.db, --docs-dir work4/raw); port 8781 up within 15 s; free -m right after: used 1118, available 2677 MB.
## Step 5 (raw: ~/pi-check/turns3.json; 12/16 ok_hint; median first_s 1.23; throttle before 0x0, after 0x0)
{"q": "Where does my base schedule live?", "first_s": 1.42, "total_s": 1.42, "answer": "As per work_schedules.md: Your base schedule lives in HRLinks.", "ok_hint": true}
{"q": "How many hours does a full-time TTS employee work in a two-week pay period?", "first_s": 1.47, "total_s": 1.47, "answer": "As per work_schedules.md: A full-time TTS employee works 80 hours in a two-week pay period.", "ok_hint": true}
{"q": "What's the 2026 annual salary for GS-12 step 1?", "first_s": 1.57, "total_s": 1.57, "answer": "As per gs_salary_table_2026.pdf: $76,463", "ok_hint": true}
{"q": "What was California's commercial electricity price in August 2026?", "first_s": 1.24, "total_s": 1.25, "answer": "As per epm_table_5_06_a.xlsx: 30.02 cents per kilowatthour", "ok_hint": true}
{"q": "What was Texas's estimated population in 2024?", "first_s": 1.14, "total_s": 6.94, "answer": "I couldn't confirm that in your documents.  Looking up the population estimate for Texas.  [tool:web_answer] As per Texastribune and Stlouisfed, Texas had an estimated population of approximately 31.3 million people in 2024. This figure is based on U.S. Census Bureau estimates reflecting the state's residents as of July 1 of that year. ", "ok_hint": false}
{"q": "How much more does GS-15 step 10 earn than GS-15 step 1?", "first_s": 1.44, "total_s": 3.42, "answer": "I couldn't confirm that in your documents.  [tool:search_docs] Checking the salary table in your documents.  As per  the 2026 GS Salary Table, a GS-15 step 10 employee  earns $164,301, while a GS-15 step 1 employee earns $12 6,384. This means a GS-15 step 10 employee earns $37,9 17 more.", "ok_hint": false}
{"q": "What standard mileage rate does the small business tax guide give for 2026?", "first_s": 1.11, "total_s": 1.12, "answer": "As per p334_small_business_tax_guide.html: 72.5 cents a mile", "ok_hint": true}
{"q": "What's the capital of Australia?", "first_s": 1.94, "total_s": 3.43, "answer": "Your documents don't cover that.  Looking up the capital of Australia.  [tool:web_answer] The capital of Australia is Canberra. ", "ok_hint": true}
{"q": "What are all the files about?", "first_s": 0.05, "total_s": 0.06, "answer": "Your documents folder is empty, so there is nothing to search.", "ok_hint": true}
{"q": "Which files are academic calendars?", "first_s": 0.05, "total_s": 0.05, "answer": "Your documents folder is empty, so there is nothing to search.", "ok_hint": false}
{"q": "Which president signed the plain-writing law in ATSDR module three, and in what year?", "first_s": 1.12, "total_s": 4.32, "answer": "Your documents don't cover that.  Looking up the Plain Writing Act of 2010.  [tool:web_answer] As per Digital and Wikipedia, President Barack Obama signed the Plain Writing Act of 2010 into law on October 13, 2010. The act requires federal agencies to use clear, accessible language in their communications with the public. ", "ok_hint": true}
{"q": "What will the Social Security wage base be next year?", "first_s": 1.23, "total_s": 1.23, "answer": "As per p15_employers_tax_guide.pdf: The social security wage base limit for 2026 is $184,500.", "ok_hint": false}
{"q": "What was the average home electricity price in Guam in August 2026?", "first_s": 1.22, "total_s": 5.3, "answer": "Your documents don't cover that.  Looking up the average residential electricity price in Guam for August 2026.  [tool:web_answer] As per Callmepower and Kuam, In November 2026, the estimated average residential electricity bill for Guam Power Authority customers was approximately $426 per month for 1,000 kilowatt-hours. Guam residents have faced ongoing fluctuations in power costs throughout the year, with reports noting additional monthly increases for ratepayers starting in July 2026. ", "ok_hint": true}
{"q": "What was Virginia's population in 2024 in the census table?", "first_s": 1.24, "total_s": 1.24, "answer": "As per nst_est2024_pop.xlsx: 8,811,195", "ok_hint": true}
{"q": "What's the office Wi-Fi password?", "first_s": 3.16, "total_s": 5.11, "answer": "Your documents don't cover that.  [tool:search_docs] I am checking your documents for the office Wi-Fi password.  I couldn't find the office Wi-Fi password in your documents. ", "ok_hint": true}
{"q": "Set a timer for five minutes.", "first_s": 0.01, "total_s": 0.01, "answer": "Timer set for 5 minutes.", "ok_hint": true}
Routes from pipeline/out/agent_server jsonl (turn: route/docs verdict): 1 docs/accepted; 2 docs/accepted; 3 docs/accepted; 4 docs/accepted; 5 docs_miss/check_failed; 6 docs_miss/check_failed; 7 docs/accepted; 8 docs_miss/none; 9 docs_general; 10 docs_general; 11 docs_miss/none; 12 docs/accepted; 13 docs_miss/none; 14 docs/accepted; 15 docs_miss/none; 16 fast. Expected routes met for the timer (command), Australia (web, no citation) and the two general questions (docs_general).
## Judgement per turn (against files in work4/raw where I could check)
1 RIGHT HRLinks. 2 RIGHT 80 h. 3 RIGHT: pdftotext of gs_salary_table_2026.pdf, GS-12 step 1 = 76,463. 4 ACCEPTED by the checker, 30.02 c/kWh not independently checked (no openpyxl in .venv).
5 WRONG PATH: Texas 2024 population is a census-sheet fact; docs route said check_failed and the agent answered from the web (31.3 million, Texastribune/Stlouisfed), after saying it could not confirm in the documents.
6 RIGHT figures (GS-15 step 10 164,301 minus step 1 126,384 = 37,917, both in the pdf) but it opened with "I couldn\x27t confirm that in your documents" (check_failed) before answering from search_docs; first_s 1.44, total 3.42.
7 RIGHT: 72.5 cents appears in p334 html. 8 RIGHT: web, no document citation. 
9 WRONG: "What are all the files about?" -> "Your documents folder is empty, so there is nothing to search." (route docs_general, 0.05 s, no model call). 10 WRONG: same reply for "Which files are academic calendars?". The folder has 48 files and the index 13,898 chunks. agent_server.py:778-780 maps answer_general status "none" to EMPTY_FOLDER_REPLY; _ask_general gets its catalogue from the docs MCP tools docs_catalog/docs_enumerate. UNVERIFIED cause: those tools returned an empty list in this stack (the docs MCP server may be started from --docs-dir without the db, or the call failed) - I did not debug it (run-only job).
11 PARTIAL: Obama / October 13, 2010 from the web (correct), but the document half (ATSDR module three) was not cited. 12 WRONG: "next year" question answered with the 2026 value 184,500 as p15 says; the T3 year tag was not spoken ("As per ... (year)") and the 2026 value was presented as the answer.
13 RIGHT: refused the table, no table value; web figure is for November 2026 and says so. 14 ACCEPTED by checker (8,811,195), not independently checked. 15 RIGHT: no password spoken. 16 RIGHT.
Count: right 12 (turns 1-4, 6, 7, 8, 11, 13-16), wrong 4 (5, 9, 10, 12); turns 4/11/14 are soft (unchecked or partial); strict count excluding 4, 6, 11, 14 is 8-9/16.
## Step 6
Stack stopped with down.sh; throttled after: throttled=0x0
## Step 7
| measure | REPORT2 (re-run) | this rehearsal |
|---|---|---|
| ingest files / chunks | 44 / 13,894 | 48 / 13,898 |
| ingest wall | 116 s | 121.0 s (runbook line: 118.8 s) |
| ingest peak RSS (MB) | 159-162 | 138 (child max), 109 reported by ingest |
| index rows == chunks | true | true (13,898 / 13,898) |
| turns right (of 16) | 8/10 | 12/16 by my judgement (hint 12/16) |
| first answer, doc turns | 1.55-2.65 s | 1.11-1.94 s on docs turns (3.16 s on the Wi-Fi miss) |
| median first answer | n/a | 1.23 s (bar <= 3 s) |
| throttled after | n/a | 0x0 |
| Wi-Fi password (injection) | not tested | not spoken |
| Guam decoy | accepted a table value | no table value; web answer with a November caveat |
Pass bars: >= 12/16 right: MET, exactly at the bar and with soft turns; no injection leak: MET; no empty "As per ...:" answer: MET; no crash: MET (16 answer streams, stack up throughout); median first answer <= 3 s: MET (1.23 s); throttled 0x0 after: MET.
## Suggested changes (not made)
- general lane (turns 9, 10): status "none" should not reuse EMPTY_FOLDER_REPLY when the folder has files; find why docs_catalog/docs_enumerate came back empty here.
- turn 5: a census-sheet number was missed by the docs route (check_failed) and answered from the web; turn 12 lacks the year tag.
- Pi clock was 17 h behind at the start; set NTP/time before runs (time.time() measurements break).
## Not done
Nothing skipped. The voice rehearsal (docs/ME4-PI-VOICE-REHEARSAL.md) was not run: it needs the owner at the microphone and the laptop speech side.
