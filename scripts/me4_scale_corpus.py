"""Synthetic corpus generator for ME4 scale probing.

Writes mixed-format corpus (pdf/csv/docx/pptx/xlsx/md/txt) with Zipf vocabulary
and one unique needle per file.  Outputs needles.json above the corpus folder.

    uv run python scripts/me4_scale_corpus.py --out /tmp/corpus --mb 40 [--zip]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

try:
    import openpyxl

    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False

try:
    from pptx import Presentation

    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

# docx is always supported via hand-built zip (no python-docx needed)
HAS_DOCX = True

TYPE_WEIGHTS = {
    "pdf": 0.40,
    "csv": 0.15,
    "docx": 0.15,
    "pptx": 0.10,
    "xlsx": 0.10,
    "md": 0.05,
    "txt": 0.05,
}
MAX_CSV_ROWS = 40_000
NEEDLE_TPL = "The access code for vault {i} is zq{i:05d}x."

# Vocabulary: common English + tech words (~300 unique)
_VOCAB = list(
    set(
        w.lower()
        for w in (  # noqa: SIM905
            "the be to of and a in that have i it for not on with he as you do at this "
            "but his by from they we say her she or an will my one all would there their "
            "what so up out if about who get which go me when make can like time no just "
            "him know take people into year your good some could them see other than then "
            "now look only come its over think also back after use two how our way even "
            "because any these give day most us is are was were been being have has had "
            "doing done does did shall should may might must will would could can "
            "report data system model user file path value function method class object "
            "index query result cache memory disk process thread lock state error "
            "configuration parameter option setting output input format type string "
            "number integer float boolean array list dictionary mapping collection "
            "search retrieve filter sort merge split chunk parse load save read write "
            "network request response protocol server client socket port address domain "
            "database table record field column row key primary foreign constraint "
            "transaction commit rollback isolation concurrency parallel async await "
            "coroutine promise future signal event handler callback loop timer "
            "performance latency throughput bandwidth optimization compression encoding "
            "decryption security authentication authorization token credential password "
            "algorithm complexity linear binary tree graph queue stack heap algorithm "
            "compute memory storage volume backup restore snapshot image container "
            "deploy release version branch merge commit diff patch build test coverage "
            "monitor metric alert dashboard log trace span context correlation "
            "infrastructure resource cluster node region zone endpoint service mesh "
            "virtual machine instance storage bucket object reference dependency module "
            "package library framework tool workflow pipeline stage trigger action "
            "schedule cron job task worker scheduler dispatcher queue buffer pool "
            "connection session token session cookie header body payload metadata "
            "schema template render view component element node attribute property "
            "method function procedure routine operation action event listener "
            "observer publisher subscriber message signal notification notification "
        ).split()
        if w.lower()
    )
)

# Pre-computed zipf weights (computed once at import)
_zipf_wts: list[float] = [1.0 / (i + 1) ** 1.05 for i in range(len(_VOCAB))]
_zipf_total: float = sum(_zipf_wts)
ZIPF_PROBS: list[float] = [w / _zipf_total for w in _zipf_wts]


def _zipf_text(rng: random.Random, chars: int) -> str:
    words: list[str] = []
    length = 0
    while length < chars + 100:
        w = rng.choices(_VOCAB, weights=ZIPF_PROBS, k=1)[0]
        words.append(w)
        length += len(w) + 1
    sentences, chunk = [], []
    for w in words:
        chunk.append(w)
        if len(chunk) >= rng.randint(8, 15):
            sentences.append(" ".join(chunk) + ".")
            chunk = []
    if chunk:
        sentences.append(" ".join(chunk) + ".")
    return "\n".join(sentences)[:chars].encode("ascii", errors="ignore").decode("ascii")


def _needle(i: int) -> str:
    return NEEDLE_TPL.format(i=i)


# File size ranges in bytes: (min, max).
# Average ~300 KB; min ~20 KB, max ~8 MB for a manageable file count.
_SIZE_RANGES: dict[str, tuple[int, int]] = {
    "pdf": (30_000, 4_000_000),
    "docx": (30_000, 4_000_000),
    "csv": (200_000, 4_000_000),
    "pptx": (200_000, 4_000_000),
    "xlsx": (200_000, 4_000_000),
    "md": (30_000, 4_000_000),
    "txt": (30_000, 4_000_000),
}


def _write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.write_text("# Document\n\n" + text, encoding="utf-8")


def _write_csv(path: Path, rng: random.Random, needle: str, target_bytes: int) -> None:
    """Write a CSV with rows to approach target_bytes.  ~100 bytes/row."""
    header = ["id", "category", "value", "description", "notes", "flag", "score"]
    count = min(MAX_CSV_ROWS - 1, max(200, target_bytes // 100))
    rows: list[list] = []
    for j in range(count):
        rows.append(
            [
                j,
                f"cat_{rng.randint(1, 50)}",
                round(rng.gauss(100, 30), 2),
                f"item_{rng.randint(1, 200)}",
                "",
                rng.choice(["A", "B", "C"]),
                round(rng.random() * 100, 4),
            ]
        )
    needle_pos = rng.randint(0, len(rows) - 1)
    rows[needle_pos][3] = needle
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _write_pdf(path: Path, text: str) -> None:
    """Minimal hand-written PDF with extractable text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = [line[:80] for line in escaped.split("\n")]
    pages: list[list[str]] = []
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) >= 50:
            pages.append(buf)
            buf = []
    if buf:
        pages.append(buf)
    if not pages:
        pages = [["(empty)"]]

    n = len(pages)
    content_texts: list[str] = []
    for plines in pages:
        y = 750.0
        parts = ["BT /F1 10 Tf"]
        for ln in plines:
            parts.append(f"50 {y:.1f} Td ({ln}) Tj")
            y -= 14
        parts.append("ET")
        content_texts.append("\n".join(parts))

    kids = " ".join(f"{2 * i + 3} 0 R" for i in range(n))
    font_obj = 2 * n + 3

    obj_strings: list[str] = []
    obj_strings.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    obj_strings.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n} >>\nendobj\n")

    for i, cs_text in enumerate(content_texts):
        pg = 2 * i + 3
        cs = 2 * i + 4
        obj_strings.append(
            f"{pg} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cs} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>\n"
            f"endobj\n"
        )
        obj_strings.append(
            f"{cs} 0 obj\n<< /Length {len(cs_text)} >>\nstream\n{cs_text}\nendstream\nendobj\n"
        )
    obj_strings.append(
        f"{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    pdf = "%PDF-1.4\n"
    offsets: list[int] = []
    for obj_str in obj_strings:
        offsets.append(len(pdf))
        pdf += obj_str
    xoff = len(pdf)
    pdf += f"xref\n0 {len(obj_strings) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(obj_strings) + 1} /Root 1 0 R >>\nstartxref\n{xoff}\n%%EOF\n"
    path.write_text(pdf, encoding="utf-8")


def _write_docx(path: Path, text: str) -> None:
    """Write a minimal hand-built docx (zipfile only) readable by load_docx."""
    wns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    paragraphs = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            escaped = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paragraphs.append(f"<w:p {wns}><w:r><w:t>{escaped}</w:t></w:r></w:p>")

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f"<w:document {wns}>"
        f"<w:body {wns}>" + "\n".join(paragraphs) + "\n</w:body></w:document>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", doc_xml)


def _write_pptx(path: Path, text: str) -> None:
    if not HAS_PPTX:
        return
    prs = Presentation()
    for p in text.split("\n")[:50]:
        if p.strip():
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "Slide"
            slide.placeholders[1].text = p[:500]
    prs.save(str(path))


def _write_xlsx(path: Path, text: str) -> None:
    if not HAS_XLSX:
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    for idx, p in enumerate(text.split("\n"), 1):
        ws.cell(row=idx, column=1, value=f"Item {idx}")
        ws.cell(row=idx, column=2, value=p[:200] if p.strip() else "")
    wb.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic ME4 corpus.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mb", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_bytes = int(args.mb * 1_000_000)
    t_start = time.perf_counter()

    # Determine active types
    missing = []
    if not HAS_XLSX:
        missing.append("xlsx")
    if not HAS_PPTX:
        missing.append("pptx")
    if missing:
        print(f"[warn] skipping types: {', '.join(missing)}", file=sys.stderr)

    active = {t: w for t, w in TYPE_WEIGHTS.items() if t not in missing}
    tw = sum(active.values())
    active = {t: w / tw for t, w in active.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    needles: list[dict[str, str]] = []
    file_counter = 0

    # Generate per type: loop until byte target is reached
    type_bytes: dict[str, int] = {t: 0 for t in active}
    type_order = sorted(active, key=lambda t: -active[t])

    for ext in type_order:
        target = int(total_bytes * active[ext])
        lo, hi = _SIZE_RANGES.get(ext, (30_000, 4_000_000))

        while type_bytes[ext] < target:
            remaining = target - type_bytes[ext]
            # Cap file size so last file doesn't overshoot too much
            max_size = min(hi, max(lo, remaining + lo))
            min_size = min(lo, max(10_000, remaining if remaining > 10_000 else lo))
            file_target = int(rng.uniform(min_size, max_size))

            needle = _needle(file_counter)
            name = f"doc_{file_counter:04d}.{ext}"
            p = args.out / name

            try:
                if ext == "csv":
                    _write_csv(p, rng, needle, file_target)
                elif ext == "pdf":
                    text_chars = max(200, file_target // 2)
                    text = _zipf_text(rng, text_chars)
                    text = needle + "\n" + text
                    _write_pdf(p, text)
                elif ext == "docx":
                    text_chars = max(200, file_target // 2)
                    text = _zipf_text(rng, text_chars)
                    text = needle + "\n" + text
                    _write_docx(p, text)
                elif ext == "pptx":
                    text_chars = max(200, file_target // 2)
                    text = _zipf_text(rng, text_chars)
                    text = needle + "\n" + text
                    _write_pptx(p, text)
                elif ext == "xlsx":
                    text_chars = max(200, file_target // 2)
                    text = _zipf_text(rng, text_chars)
                    text = needle + "\n" + text
                    _write_xlsx(p, text)
                elif ext == "txt":
                    text = _zipf_text(rng, max(200, file_target))
                    text = needle + "\n" + text
                    _write_txt(p, text)
                elif ext == "md":
                    text = _zipf_text(rng, max(200, file_target))
                    text = needle + "\n" + text
                    _write_md(p, text)
            except Exception as exc:
                print(f"[warn] failed {p.name}: {exc}", file=sys.stderr)
                if p.exists():
                    p.unlink()
                continue

            if p.exists():
                written = p.stat().st_size
                type_bytes[ext] += written
                needles.append(
                    {
                        "file": name,
                        "query": f"zq{file_counter:05d}x",
                        "answer": needle,
                    }
                )
                file_counter += 1

    # Shuffle needles so counter != positional info
    rng.shuffle(needles)
    np = args.out.parent / "needles.json"
    np.write_text(json.dumps(needles, indent=2) + "\n", encoding="utf-8")

    written_files = [f for f in args.out.glob("*") if f.is_file()]
    tw_actual = sum(f.stat().st_size for f in written_files)
    elapsed = time.perf_counter() - t_start
    mb_per_s = (tw_actual / 1_000_000) / elapsed if elapsed > 0 else 0

    tw_mb = tw_actual / 1_000_000
    print(
        f"ok {len(written_files)} files {tw_mb:.1f} MB "
        f"in {elapsed:.1f}s ({mb_per_s:.1f} MB/s) -> {np}"
    )

    # Print type breakdown
    for ext in type_order:
        actual = type_bytes[ext] / 1_000_000
        tgt = total_bytes * active[ext] / 1_000_000
        cnt = sum(1 for f in written_files if f.suffix == f".{ext}")
        print(f"  {ext}: {actual:.1f} MB target={tgt:.1f} MB ({cnt} files)", file=sys.stderr)

    if args.zip:
        zp = args.out.parent / f"{args.out.name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(written_files):
                zf.write(p, arcname=p.name)
        print(f"zip {zp} ({zp.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
