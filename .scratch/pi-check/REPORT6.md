# Pi run 6: standin_c2 (387 files, unseen set) on d53ad6b, 2026-11-27

Code: d53ad6b (247 compound, 251-253 ingest fixes, POSTPROC_VERSION 3). Zip sha256 3c7759eb... (HF `martinnavs/me4-pi-package`).

## Ingest (Pi 4, 4 workers)
- 386 extracted (1 junk skipped by the zip step), 371 ingested: 370 ok, 1 failed (edge/corrupt.pdf, recorded "unreadable"), 11 unsupported recorded with 0 chunks (8 worldbank/raw json, edge .odt/.eml/.json), 4 more junk ignored.
- 13,003 chunks, wall 271 s, peak 99 MB. Integrity check ok. DB + raw 142 MB.
- Edge cases: mixed PDF kept 2 text pages; blank PDF "scanned: 4 pages without text"; xls roster keeps blank cells (Floor 4 for Ines Karrow present in the chunk); duplicate README names and the unicode path ingested.

## 35 questions (standin_c2_questions.jsonl), run on the Pi against the live agent
21 pass, 6 partial, 8 fail. First answer median 1.58 s, max 3.93 s (all 35 in run35.json).

| group | result |
|---|---|
| mixed PDF (2) | 2/2 pass (251 works) |
| xls blank cell | Mirela Tonk badge pass; Ines Karrow floor FAIL (docs "none", fell to the web: "could not find a person") |
| duplicate README | pass |
| unicode folder | FAIL (check_failed, web answer about Cebu City Council) |
| unsupported .odt | FAIL: web guess ("Harbour City ... 10 AM to 9 PM") instead of "I couldn't read that file" |
| World Bank tables (18) | 13 pass from the documents; 5 went docs_miss -> web: gdp Indonesia 2013 (web 912.52 bn, right but unsourced), Nigeria co2 (web 0.60, table 0.621), India co2 (1.6, table 1.576), Ethiopia electricity (web ~100, table 93.6: wrong), Italy electricity (web 290 bn kWh, table 5041 per capita: different quantity) |
| traps (4) | Atlantis ok; injection refused ("The documents don't say"); Wi-Fi and Brindle: "couldn't confirm with a search, so I won't guess" (honest, but says nothing about the documents) |
| pleasantries (2) | pass, 0.6-0.7 s, no document call |
| doc + web (4) | all four went to the web only; the table/roster half is missing in dw-1 (Survey), dw-3 (Japan table value), dw-4 (Kenya table value); dw-2 gave the smallest country but not its capital. 247 did not take effect on these |

## Findings
1. CSV with a title line above the header: every csv chunk has `col_N` as column names (`life_expectancy`, `electric_power_use`, `gdp_current_usd`: 15/29/29 chunks, all `col_2`; the year names appear in only 1-3 chunks). The xlsx copies keep the years (0 col_N). The 5 table misses all retrieved the right data file (csv/xlsx/xls in sources) but the answer check failed. The year-less csv chunks are the likely cause; not proven. Reader fix candidate: skip a one-cell title row when taking the header (`readers.py`, csv path).
2. The docs-first check rejects answers whose evidence is present (Ines Karrow, committee, 5 tables): `docs: check_failed` or `none`. Falls back to the web, where the answer may be wrong (Ethiopia, Italy).
3. Unsupported files: the question names no file, so the spoken "N files are in a format I can't read" never triggers; the web guesses instead.
4. 247 compound (document + web) did not fire on the 4 dw questions (route docs_miss, web_answer); not investigated why.
5. Operations: the git archive deploy resets the scripts' execute bit; the agent crash-looped (restart counter 75) until `chmod +x scripts/stack/*.sh`. The laptop cannot reach the Pi's IPv4 after the share was torn down because the laptop's masquerade rule is still active; the driver was run on the Pi instead.
