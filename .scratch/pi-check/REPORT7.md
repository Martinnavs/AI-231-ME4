# Pi run 7: standin_c2 re-ingest on 1421c7c (csv title-line fix 254), 2026-11-27

Same 35 questions as REPORT6, fresh `work4` (db + raw cleared, re-ingested from the same zip). Raw answers: `run35b.json` (run 6: `run35.json`).

## Ingest
371 files, 370 ok / 1 failed (corrupt.pdf) / 11 unsupported recorded; 12,945 chunks (run 6: 13,003); wall 271 s, peak 99 MB; integrity ok. The three csv tables checked (life_expectancy, electric_power_use, gdp_current_usd) now have 0 `col_2` chunks and the year in every chunk (before: all chunks `col_N`, year in 1-3). `LEGACY_PARSE_VERSION` 3 was enough here, but I cleared the cache folder too, so I did not test whether an old `.rag_cache` would have been reused.

## Result: 27 pass, 4 partial, 4 fail (run 6: 21 / 6 / 8). First answer median 1.33 s, max 3.79 s (run 6: 1.58 / 3.93).
- Fixed: Ines Karrow floor (xls), unicode-folder committee (12 March 14:00), Italy electricity, Nigeria and India CO2, Indonesia GDP. Tables 17 of 18 pass from the documents.
- Trap-4 (injection) now refuses in words and offers to look for specific content.
- Still wrong:
  - Ethiopia electricity 2020: docs_miss -> web, which gave 10.93 billion kWh (the table has 93.6 kWh per capita). The csv/xlsx/xls were retrieved in run 6; cause not found.
  - dw-2 (capital of the smallest-2020-population country in the table): **false accept**: docs "accepted" the wiki page Neuquen.html and spoke "Neuquén". Wrong answer from a checked document path.
  - dw-1 (Alder Voss + Japan PM): the web half answered and denied the roster person ("operates his own business"); the roster says Survey. The document half is missing.
  - dw-3 (people in Japan): web 122.1 million, the table value and year are not mentioned.
- Partial: dw-4 (Kenya: web for both halves, table life expectancy not used), trap-1 and trap-2 ("couldn't confirm that with a search", says nothing about the documents), edge-unsupported-1 (no guess now, but "with a search", not "I couldn't read that file").
- 247 compound still did not take effect on the four doc+web questions.

## Cache check (stale `.rag_cache` after the fixes), on the Pi
A 5-file folder (3 world-bank csv with a title line, the mixed PDF, the xls roster): ingest with old code to build `.rag_cache`, then ingest the same folder with HEAD (1421c7c), cache kept.
- Old cache from d53ad6b (before 254): csv chunks 29/29/15 with `col_2`; HEAD over that cache: 15/15/11, 0 `col_2`. The csv text is re-parsed, not reused.
- Old cache from c484bde (before 251-253 and POSTPROC 3): the mixed PDF has no chunks; HEAD over that cache gives its 2 text pages; csv fixed as above. HEAD over the old cache is chunk-for-chunk equal (sha256) to HEAD with the cache removed.
- Live corpus warm re-ingest (cache from the run-7 ingest, scratch db): 371 files, 12,945 chunks, identical chunk hashes to the live `docs.db`; 137 s warm vs 271 s cold.
So a Pi folder ingested before these commits does not need its cache cleared. Not tested: a cache made by the zip-ingest path of the very first versions, and the legacy (non-canonical) cache of other file kinds (pdf/docx).
