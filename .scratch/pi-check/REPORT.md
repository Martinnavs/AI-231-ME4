# ME4 Pi check report
Archive of qwen/ingest commit: 47817d70b200d2a10c748e0864896b4c69d2f6a4 (unpacked via git archive; no git log on Pi)

## Task 1
(ME3 stack removed by owner request before this run; cleared uv/pip/mozilla caches)
```
aarch64
4
               total        used        free      shared  buff/cache   available
Mem:            3796         517        2831          83         599        3279
Swap:           2047           0        2047
swapon: not installed; swap per free above
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           1.9G  4.0K  1.9G   1% /tmp
/dev/mmcblk0p7   14G  9.2G  4.1G  70% /
throttled=0x0
Python 3.13.5
```

## Task 2
sudo -n true: FAILS (password required) -> tesseract: no sudo
rapidocr: not installed (ModuleNotFoundError)

## Task 2 (cont.)
uv sync --group pi-rag: completed, 68 packages, no failures (log: sync.log). rapidocr: not installed. tesseract: no sudo.

## Task 3 (zip: 44 files, 65.6 MB, no .rag_cache)
```
{
 "facts": {
  "machine": "aarch64",
  "cpu": "Raspberry Pi 4 Model B Rev 1.2",
  "mem_mb": {
   "MemTotal": 3796,
   "MemAvailable": 3257,
   "SwapTotal": 2047,
   "SwapFree": 2047
  },
  "disk_free_mb": 3726,
  "throttled": "throttled=0x0",
  "python": "3.12.15"
 },
 "ingest": {
  "exit": 0,
  "wall_s": 87.2,
  "parent_peak_mb": 162,
  "last_line": "ingest: 44 files, 44 ok, 0 failed, 0 skipped (disk/time), 13894 chunks, 162 MB RSS, 0.0s",
  "stderr_tail": ""
 },
 "integrity": {
  "files": {
   "ok": 44
  },
  "chunks": 13894,
  "search_index_rows": 13894,
  "index_matches_chunks": true,
  "images": {
   "pending": 133
  }
 },
 "queries": {
  "median_s": 1.23,
  "max_s": 1.41,
  "rows": [
   {
    "q": "What is this document about?",
    "s": 1.23,
    "files": [
     "csus/syllabus_template_guide_2025.docx",
     "ietf/rfc8259_json.txt",
     "uoregon/syllabus_starter_template.docx"
    ]
   },
   {
    "q": "How many pages or rows does the largest file have?",
    "s": 1.41,
    "files": [
     "epa/water_quality_module1.pptx",
     "gsa-tts/work_schedules.md",
     "irs/i1040gi_form1040_instructions.pdf"
    ]
   },
   {
    "q": "Which file mentions a deadline or a date?",
    "s": 1.2,
    "files": [
     "nyu/liberal_studies_academic_calendar.html",
     "uh/academic_calendar_2026_2027_fall.html",
     "uoregon/syllabus_starter_template.docx"
    ]
   },
   {
    "q": "What is the total or the maximum value in the spreadsheet?",
    "s": 1.06,
    "files": [
     "epa/water_quality_module1.pptx",
     "gsa-tts/work_schedules.md",
     "ons/capex_population_sample_counts.xlsx"
    ]
   },
   {
    "q": "Who is the author or the contact person?",
    "s": 1.3,
    "files": [
     "epa/water_quality_module1.pptx",
     "ietf/rfc8259_json.txt",
     "irs/p334_small_business_tax_guide.html"
    ]
   }
  ]
 },
 "ocr": {
  "engine": null,
  "note": "no rapidocr models and no tesseract binary"
 }
}
```

## Task 4 (re-run, same work dir)
wall_s 101.0 (first 87.2), parent_peak_mb 159, 13894 chunks, search_index_rows 13894, index_matches_chunks true, files ok 44, images pending 133 (all unchanged). Re-run was NOT faster than first run.

## Task 5
text stack not startable: no router GGUF file on the Pi (find ~ -name *.gguf found none; no GEMINI env var; a key file exists under ~/.secrets but was not opened). Skipped.

## Task 6
Skipped (not authorised).

## Task 7: differences from laptop >3x
- ingest wall: 87.2 s vs 15-18 s laptop = 4.8-5.8x (re-run 101 s = 5.6-6.7x)
- queries median: 1.23 s vs 0.4 s = 3.1x
- parent peak RSS 162 MB vs 100 MB = 1.6x (not >3x)
- OCR: no engine, not measured

## Suggested changes
- Re-run ingest took longer than first run: resume/skip-unchanged does not appear to skip work in me4_pi_check.py (or the script wipes the work dir).
- OCR figure needs tesseract (sudo) or RapidOCR models.

## Addendum: OCR with tesseract 5.5.0 (installed by owner)
Re-run into fresh work dir (pi-check-ocr.json): ingest 77.5 s, parent peak 159 MB, 13894 chunks = 13894 index rows, queries median 1.25 s.
ocr: engine available (tesseract), 30 images, total 16.7 s, median 0.19 s, p90 1.4 s, 7 of 30 had text, peak RSS 87 MB.
Laptop (RapidOCR x86): median 0.72 s, p90 7.5 s. Pi tesseract is faster per image in this sample but engines differ and only 7/30 images had text.

## Task 5 (replaces the earlier "not startable"): text stack, five timed turns
Setup deviations (none edit the repo): no router GGUF/llama on the Pi so NO_ROUTER=1 (Gemini answers every turn, no local router); pipeline venv synced (uv sync in pipeline/); fastmcp==2.14.7 installed into the root .venv with uv pip; random token created at ~/.config/lan_speech_token; scripts/stack/up.sh is not executable in the archive so it was run with bash; "--docs-first" is NOT an agent_server flag on this branch, docs-first is the env var VOICE_AGENT_DOCS_FIRST=1 (docs/ME4-NEXT-PLAN.md) plus --docs-dir ~/pi-check/work2/raw --docs-retriever fts5.
IPv6 finding: the Pi has a broken IPv6 route; a TCP connect to generativelanguage.googleapis.com:443 takes 8.2 s (IPv4 0.09 s). OnlineProbe (agent_server.py:160, 2 s connect, no IPv4 pinning) therefore marks the server offline and every turn answered "I cant reach the internet right now. Worked around with an IPv4-only sitecustomize shim in ~/pi-check/shim (PYTHONPATH), not in the repo.
Timed with a small websocket driver (~/pi-check/turns.py, uses text_chat.client.AgentClient; the terminal UI is not scriptable). Seconds to first answer text frame, which for doc questions is the searching filler sentence; total = last frame.
| # | question | first | total | answer | right? |
|---|---|---|---|---|---|
| 1 | GSA TTS handbook: what is a base schedule, where does it live? | 1.68 | 2.03 | defined it correctly from gsa-tts/work_schedules.md; said the documents do not say where it lives | PARTIAL: the file says it lives in HRLinks; the agent missed that (a 1st run gave the same miss) |
| 2 | US population 2024 in the census file | 2.47 | 3.10 | 340,110,988 | RIGHT (csv row United States POPESTIMATE2024 = 340110988) |
| 3 | What is the weather today? | 0.88 | 0.91 | weather tool: Cebu City drizzle 25-31 C | RIGHT (not from documents) |
| 4 | GSA TTS pets policy (not in the files) | 2.62 | 3.20 | The documents dont say." | RIGHT (grep for pets finds nothing) |
| 5 | Which file did that come from? (right after Q1) | 0.89 | 0.90 | gsa-tts/work_schedules.md | RIGHT |
Memory: 2,556 MB available with the stack up; no swap use observed; disk 2.5 GB free after the extra uv syncs.
Note: running the pipeline with uv run re-synced extra packages (pandas, sherpa-onnx, onnxruntime, ~90 packages) into pipeline/.venv, using about 1 GB of disk.
