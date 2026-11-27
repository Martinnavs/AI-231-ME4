# ME4 release manifest: freeze candidate 2026-11-27

**Code:** `1421c7c` (tag `me4-rc-2026-11-27`, branch `origin/qwen/ingest`). Later commits on the branch are docs/results only.
**Deploy:** `git archive me4-rc-2026-11-27` onto the Pi and the laptop (scripts are executable in git since 1421c7c), `--hedge -1` in `AGENT_ARGS` (`config/pi.env`), `systemctl --user restart pi-agent`. Build the ASR vocabulary from the real corpus on the laptop: `scripts/make_entities.py <corpus> --merge config/entities.json -o config/entities-docs.json`, start the speech server with `--entities config/entities-docs.json`. A folder ingested before this release needs no cache clearing (`POSTPROC_VERSION` 3, `LEGACY_PARSE_VERSION` 3; verified on the Pi in REPORT7).

## What is in it
Fixes 234-247 (router, chain, docs-miss path, calc, next-year guard, web directness check, compound split), ingest fixes 251 (mixed PDF keeps text pages), 252 (unsupported files recorded, junk ignored, spoken once as a count), 253 (`.xls` keeps columns/datemode/one-line cells), 254 (csv title line is not the header), executable scripts. Flags: `COMPOUND=True` (247), `DIRECTNESS=True` (245), `SUFFICIENCY` does not exist on this branch (experiment branch only), `REQUIRE_QUOTES=False`, `ROW_KEY_CHECK=False`.

## Evidence
- Offline gates (fresh ingests, by row id): zero lost rows, zero gained rows, false accepts unchanged (golden 3, A 1, B 2, nondoc25 5), decoys rejected.
- Suites: pipeline 1577 passed, 1 xfailed; root 2406 passed, 36 skipped, 5 xfailed; the only failures are `test_llm_echo_is_accepted_without_documents` (baseline) and, in the run that predates the fix, a version pin (fixed).
- Pi (4 GB Pi 4) on the 48-file corpus, `c484bde`: 27/40 pass, chain capture 0/68, median first answer 3.4 s.
- Pi on the unseen 387-file stand-in (`standin_c2.zip`, sha256 prefix `3c7759eb`), `1421c7c`: ingest 271 s cold / 137 s warm, 99 MB peak, 12,945 chunks; 35 questions: 27 pass, 4 partial, 4 fail, median first answer 1.33 s, max 3.79 s.
- Generality: 300 random `.xls` (old code misplaced values in 297, new 0), 120 random PDFs (old lost text pages in 47, new 0), random folder tree (all supported files identical, all unsupported recorded, junk not recorded).

## Known limits shipped
1. Document + web questions return only the web half unless the question names the documents (247 fires only on a docs miss that references documents). Stand-in: 0/4; golden-v2 sample: 5/10.
2. One false accept seen on a compound question (a wiki page accepted for "capital of the country in the population table with the smallest 2020 value"). The document path cannot tell that the table premise was never answered.
3. Honest misses say "couldn't confirm that with a search" and say nothing about the documents; a question about an unreadable file type (.odt, .eml) gets that wording, not "I couldn't read that file".
4. Ethiopia electricity (table: 93.6 kWh per capita) goes docs_miss -> web on the Pi; cause not found (offline with Qwen it is accepted).
5. d-10 wrong row; calc rows may still refuse live; temporal "this month" rows; year-tag card wiring not done; "I can't reach the internet" turns on a flapping network (test the real network).
6. The 160 MB / 509-file ingest on the Pi has not been run (the 387-file stand-in has: 271 s).

## Rollback chain
`1421c7c` -> `7ad8101` (fixes 240-247 without the ingest fixes) -> `8dbc4e3` (RC-A, 234) -> RC-B -> `dfc068f` (RC-0). Each fix is its own merge and reverts alone.

## Post-test queue
Wall-clock budget (A3), real-model eval harness (Qwen answerer + judge), date/card wiring (A1+A6), prompt additions (A4), one bounded repair (A2), sufficiency check (branch `exp/sufficiency`, set 249), correction turns (A5), zip staging bounds, dbindex lock, `file_has` via FTS, a corpus-derived ASR vocabulary, deep-research harness behind a flag.
