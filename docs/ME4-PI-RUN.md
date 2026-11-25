# ME4 on the Pi: what to run and what to send back (2026-11-25)

Goal: replace our laptop estimates with measurements. Nothing here needs sudo except the optional OCR install, and it stops no service.

## 1. Get the code (once)
    git fetch origin && git checkout qwen/ingest && git pull
    uv sync --group pi-rag          # or the project's usual Pi sync; python3.12

## 2. Optional OCR engines (skip if short of time; the check reports "no engine")
    sudo apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng   # ~25 MB
    # RapidOCR needs the `rapidocr` package and the model files in assets/ocr/
    # (en_PP-OCRv4_mobile_rec.onnx, en_dict.txt, ch_PP-OCRv4_det_infer.onnx; ~14 MB). Not bundled yet.

## 3. The one command (about 5-10 min for the 44-file zip)
    PYTHONPATH=src uv run python scripts/me4_pi_check.py /path/to/ipo44.zip \
        --work /tmp/pi-check --out ~/pi-check.json

It ingests the zip, checks the search index has as many rows as chunks (`index_matches_chunks` must be true),
times 5 questions against the index, and, if an engine is installed, times OCR on 30 images (scans first).
Send back `~/pi-check.json`. Key fields: `ingest.wall_s`, `ingest.parent_peak_mb`, `integrity.index_matches_chunks`,
`queries.median_s`, `ocr.median_s`, `ocr.total_s`, `facts.mem_mb`, `facts.throttled`.

## 4. What we compare it with (laptop, 44 files / 13,894 chunks)
ingest 15-18 s, parent 100 MB; 5 file_context queries median 0.4 s; OCR (x86, RapidOCR) median 0.72 s, p90 7.5 s per image.
Laptop-to-Pi slowdown seen before: 9.8-14.3x (Python indexing), 2.4-3x (pdfium).

## 5. If you have 10 more minutes: 5 spoken turns end to end
Start the agent server with `--docs-first`, ask 5 questions (2 from the documents, 1 weather/time, 1 not in the documents, 1 follow-up
"which file?"), and note the time from end of your speech to first sound for each.
