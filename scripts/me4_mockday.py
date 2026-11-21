"""Build the mock-day zip: files copied from ipo-2026 plus five hand-made office files.

uv run --no-sync python scripts/me4_mockday.py --picks tests/fixtures/me4/mockday_files.txt \
    --ipo-raw ~/rag-testsets/ipo-2026/raw --out /tmp/mockday.zip
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import openpyxl
from pptx import Presentation

CSV = (
    "name,team,city,desk\n"
    "Ana Reyes,Platform,Manila,D-114\n"
    "Ben Cruz,Platform,Cebu,D-220\n"
    "Carla Diaz,Finance,Manila,D-307\n"
    "Dev Patel,Finance,Pune,D-308\n"
    "Eli Tan,Support,Davao,D-415\n"
)
BUDGET = [
    ("item", "q1", "q2", "q3"),
    ("Cloud hosting", 1200, 1350, 1500),
    ("Travel", 800, 650, 900),
    ("Training", 300, 450, 500),
    ("Equipment", 2000, 400, 700),
]
SLIDES = [
    ("Kickoff plan", "Project Heron starts on 3 March and ships on 28 June."),
    ("Risks", "The main risk is the vendor delay of two weeks for the sensor boards."),
]
MEMO = [
    "Team memo",
    "The quarterly all-hands moves to Thursday 14 May in room Tanager.",
    "Please book travel at least ten days before the event.",
]
NOTES = "Site visit notes. The Cebu office has 42 desks and the lease ends on 31 January 2027.\n"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx_bytes(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-'
            'package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.document.main+xml"/></Types>'
        ),
        "_rels/.rels": (
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>"
        ),
        "word/document.xml": (
            f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>{body}</w:body></w:document>'
        ),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in parts.items():
            zf.writestr(name, text)
    return buf.getvalue()


def xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in BUDGET:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def pptx_bytes() -> bytes:
    prs = Presentation()
    for title, body in SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def handmade_files() -> dict[str, bytes]:
    return {
        "handmade/staff_directory.csv": CSV.encode(),
        "handmade/q3_budget.xlsx": xlsx_bytes(),
        "handmade/kickoff.pptx": pptx_bytes(),
        "handmade/team_memo.docx": docx_bytes(MEMO),
        "handmade/site_visit_notes.txt": NOTES.encode(),
    }


def build_mock_zip(out_zip: Path, ipo_raw: Path, picks: list[str]) -> list[str]:
    """Write ``out_zip`` from the picked ipo-2026 files plus the hand-made ones; return names."""
    names: list[str] = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in picks:
            zf.write(ipo_raw / rel, f"ipo-2026/{rel}")
            names.append(f"ipo-2026/{rel}")
        for name, data in handmade_files().items():
            zf.writestr(name, data)
            names.append(name)
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--ipo-raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    picks = [line.strip() for line in args.picks.read_text().splitlines() if line.strip()]
    names = build_mock_zip(args.out, args.ipo_raw, picks)
    print(f"wrote {args.out} with {len(names)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
