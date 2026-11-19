"""Generate office binary fixtures (``team_budget.xlsx`` and ``site_review.pptx``).

Run from the repository root with::

    uv run python tests/fixtures/make_rag_office_fixtures.py
"""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl  # type: ignore[import-untyped]
import pptx  # type: ignore[import-untyped]
from pptx.util import Inches  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_office"
XLSX_NAME = "team_budget.xlsx"
PPTX_NAME = "site_review.pptx"


def build_team_budget_xlsx_bytes() -> bytes:
    """Build a 2-sheet numeric workbook with a merged header cell on the first sheet."""
    wb = openpyxl.Workbook()

    ws_ops = wb.active
    assert ws_ops is not None
    ws_ops.title = "Operations"
    ws_ops["A1"] = "Operations Quarterly Allocation"
    ws_ops.merge_cells("A1:D1")
    ws_ops.append(["Department", "Headcount", "Travel USD", "Equipment USD"])
    ws_ops.append(["Field Telemetry", 6, 4200, 11800])
    ws_ops.append(["Water Systems", 4, 2900, 9600])
    ws_ops.append(["Solar Maintenance", 5, 3150, 14400])

    ws_hw = wb.create_sheet("Hardware")
    ws_hw.append(["Item Code", "Unit Cost USD", "Quantity", "Total USD"])
    ws_hw.append(["SENS-LORA-09", 185, 24, 4440])
    ws_hw.append(["BATT-LIFEPO-48", 620, 12, 7440])
    ws_hw.append(["VALVE-SOL-12", 140, 18, 2520])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_site_review_pptx_bytes() -> bytes:
    """Build a 2-slide presentation with a table on slide 1 and prose on slide 2."""
    prs = pptx.Presentation()

    slide1 = prs.slides.add_slide(prs.slide_layouts[5])
    assert slide1.shapes.title is not None
    slide1.shapes.title.text = "Station Inspection Metrics"
    table_shape = slide1.shapes.add_table(
        4,
        4,
        Inches(0.8),
        Inches(1.8),
        Inches(8.2),
        Inches(2.4),
    )
    table = table_shape.table
    rows = [
        ["Station", "Elevation m", "Battery V", "Signal dBm"],
        ["North Ridge", "412", "13.4", "-84"],
        ["East Reservoir", "268", "12.9", "-91"],
        ["South Orchard", "154", "13.1", "-79"],
    ]
    for r_idx, row_vals in enumerate(rows):
        for c_idx, val in enumerate(row_vals):
            table.cell(r_idx, c_idx).text = val

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    assert slide2.shapes.title is not None
    slide2.shapes.title.text = "Maintenance Field Findings"
    slide2.placeholders[1].text = (
        "Solar charge controllers at North Ridge passed load testing after firmware "
        "update v2.4. Technicians replaced two corroded conduit gaskets on the East "
        "Reservoir junction box and verified manual bypass valves operate smoothly."
    )

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def generate_fixtures(output_dir: Path = FIXTURES_DIR) -> tuple[Path, Path]:
    """Write ``team_budget.xlsx`` and ``site_review.pptx`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = output_dir / XLSX_NAME
    pptx_path = output_dir / PPTX_NAME
    xlsx_path.write_bytes(build_team_budget_xlsx_bytes())
    pptx_path.write_bytes(build_site_review_pptx_bytes())
    return xlsx_path, pptx_path


def main() -> None:
    """CLI entry point for generating office RAG fixtures."""
    xlsx_path, pptx_path = generate_fixtures()
    print(f"Wrote {xlsx_path} and {pptx_path}")


if __name__ == "__main__":
    main()
