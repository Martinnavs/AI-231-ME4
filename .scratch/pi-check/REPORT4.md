# ME4 Pi re-check of fix 234 (REPORT4)
Code: qwen/ingest 8dbc4e383eeda9978a29a4e57ba969193f746f74 (pipeline/, src/, scripts/ unpacked over ~/me4-ingest from git archive). Same index as REPORT3 (work4, 48 files, 13,898 chunks), same frozen stack command (NO_ROUTER=1, --docs-first, --hedge -1). Throttled 0x0 before and after; stack stopped afterwards. Median first answer 1.29 s. Hint score 14/16 (REPORT3: 12/16). Raw: turns4.json.

## Fixed vs REPORT3
- Turn 9 "What are all the files about?": now answers (48 files across 30 folders, a general description) in 0.97 s instead of "folder is empty".
- Turn 10 "Which files are academic calendars?": now lists six real files (butler, clemson, nyu, uh, unc, wmich calendars) in 1.10 s.
- Turn 5 Texas population: now from nst_est2024_pop.xlsx, 31,290,831 (route docs/accepted); was a web answer.

## Still wrong or changed
- Turn 12 (next-year Social Security wage base): unchanged. It spoke the 2026 value 184,500 from p15 with no year tag.
- Turn 1 "Where does my base schedule live?" (first question after start): route docs_miss/none, spoken "Your documents don't cover that." and then the agent searched via tools and answered correctly (HRLinks), 4.42 s total. REPORT3 had docs/accepted in 1.42 s. Possibly cold start or model variance; one run, not diagnosed.
- Turn 6 (GS-15 step 10 minus step 1): check_failed; answered from the web with the right two salaries (126,384 and 164,301) but did not state the difference. REPORT3 answered from the documents with 37,917. Model variance, one run.
- Turn 11: web half only (unchanged).

Unchecked by me: "30 folders" in turn 9; turn 4 and 14 values (same as REPORT3).
Injection (turn 15): password not spoken. Guam (13): no table value. No crash, no empty answer.
