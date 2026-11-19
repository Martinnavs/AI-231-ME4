"""Generate parser regression fixtures and ``expected.json`` manifest.

Run from the repository root with::

    uv run python tests/fixtures/make_rag_regress_fixtures.py
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl  # type: ignore[import-untyped]

from voice_agent.rag import canonical, metrics
from voice_agent.rag.docindex import DocIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_regress"
PDF_NAME = "two_column.pdf"
XLSX_NAME = "merged_cells.xlsx"
EXPECTED_NAME = "expected.json"

CONVERTER_PACKAGES: tuple[str, ...] = (
    "markitdown",
    "pypdf",
    "pdfminer.six",
    "openpyxl",
    "pandas",
    "python-pptx",
    "lxml",
)

ARM_NAMES: tuple[str, ...] = ("legacy", "pypdf", "markitdown", "direct")
ARM_CONVERTERS: dict[str, Mapping[str, str]] = {"direct": {"xlsx": "direct"}}
ARM_FIXTURES: dict[str, tuple[str, ...]] = {"direct": (XLSX_NAME,)}
DEFAULT_ARM_FIXTURES: tuple[str, ...] = (PDF_NAME, XLSX_NAME)

LEFT_COLUMN_SENTENCE = "Alpha sensors measure 42 highland plots every calm morning accurately."
RIGHT_COLUMN_SENTENCE = "Bravo relays transmit 108 telemetry packets over encrypted mesh links."

MERGED_TITLE = "Quarterly Field Report"
VERTICALLY_MERGED_LABEL = "North Basin"
SHEET_NAME = "Metrics"
VERSION_MISMATCH_MESSAGE = "converter version changed: rerun and review expected.json"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_two_column_pdf_bytes() -> bytes:
    """Build a single-page PDF with a raw two-column content stream."""
    # Emit lines in row-major visual order so layout-unaware parsers interleave
    # the two columns while column-aware extractors reconstruct each sentence.
    ops = [
        "BT",
        "/F1 11 Tf",
        f"1 0 0 1 72 720 Tm ({_pdf_escape('Alpha sensors measure 42')}) Tj",
        f"1 0 0 1 340 720 Tm ({_pdf_escape('Bravo relays transmit 108')}) Tj",
        f"1 0 0 1 72 702 Tm ({_pdf_escape('highland plots every calm')}) Tj",
        f"1 0 0 1 340 702 Tm ({_pdf_escape('telemetry packets over')}) Tj",
        f"1 0 0 1 72 684 Tm ({_pdf_escape('morning accurately.')}) Tj",
        f"1 0 0 1 340 684 Tm ({_pdf_escape('encrypted mesh links.')}) Tj",
        "ET",
    ]
    stream = "\n".join(ops).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"
        ),
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    out.extend(trailer.encode("ascii"))
    return bytes(out)


def build_merged_cells_xlsx_bytes() -> bytes:
    """Build an xlsx workbook with A1:C1 merged title, vertical merge, and percent cells."""
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = SHEET_NAME

    ws["A1"] = MERGED_TITLE
    ws.merge_cells("A1:C1")

    ws.append(["Region", "Site", "Uptime"])
    ws.append([VERTICALLY_MERGED_LABEL, "Ridge One", 0.15])
    ws.append([None, "Valley Two", 0.85])
    ws.merge_cells("A3:A4")

    ws["C3"].number_format = "0%"
    ws["C4"].number_format = "0%"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def get_installed_converter_versions(
    packages: Sequence[str] = CONVERTER_PACKAGES,
) -> dict[str, str]:
    """Return installed distribution versions for the converter stack."""
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


def verify_converter_versions(expected_versions: Mapping[str, str]) -> None:
    """Raise ``AssertionError`` when installed converter versions differ."""
    installed = get_installed_converter_versions(tuple(expected_versions.keys()))
    if installed != dict(expected_versions):
        raise AssertionError(VERSION_MISMATCH_MESSAGE)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def evaluate_arm_outcomes(
    fixture_dir: Path,
    arm_name: str,
) -> dict[str, Any]:
    """Index ``fixture_dir`` under ``arm_name`` and measure regression outcomes."""
    converters: Mapping[str, str] | None = (
        dict(ARM_CONVERTERS[arm_name])
        if arm_name in ARM_CONVERTERS
        else metrics.resolve_converters_arg(arm_name)
    )
    fixture_names = ARM_FIXTURES.get(arm_name, DEFAULT_ARM_FIXTURES)
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        for name in fixture_names:
            shutil.copy2(fixture_dir / name, work_dir / name)

        idx = DocIndex(work_dir, converters=converters, canonical_cache_dir=None)
        idx.refresh()

        records_by_name = {rec.name: rec for rec in idx.records()}
        outcomes: dict[str, Any] = {}

        if PDF_NAME in fixture_names:
            pdf_chunks = [c for c in idx.chunks if c.source == PDF_NAME]
            pdf_joined = "\n".join(c.text for c in pdf_chunks)
            pdf_norm = _normalize_ws(pdf_joined)
            left_intact = LEFT_COLUMN_SENTENCE in pdf_norm
            right_intact = RIGHT_COLUMN_SENTENCE in pdf_norm
            outcomes[PDF_NAME] = {
                "status": records_by_name[PDF_NAME].status,
                "left_sentence_intact": left_intact,
                "right_sentence_intact": right_intact,
                "sentence_intact": left_intact and right_intact,
                "digit_runs": canonical.digit_runs(pdf_joined),
            }

        if XLSX_NAME in fixture_names:
            xlsx_chunks = [c for c in idx.chunks if c.source == XLSX_NAME]
            xlsx_joined = "\n".join(c.text for c in xlsx_chunks)
            title_count = xlsx_joined.count(MERGED_TITLE)
            vertical_count = xlsx_joined.count(VERTICALLY_MERGED_LABEL)
            outcomes[XLSX_NAME] = {
                "status": records_by_name[XLSX_NAME].status,
                "merged_title_count": title_count,
                "vertical_merged_count": vertical_count,
                "merged_value_present_once": {
                    "a1_c1_title": title_count == 1,
                    "vertical_label": vertical_count == 1,
                },
                "digit_runs": canonical.digit_runs(xlsx_joined),
                "sheet_locator": (
                    bool(xlsx_chunks)
                    and all(
                        c.sheet == SHEET_NAME and c.section == f"Sheet {SHEET_NAME}"
                        for c in xlsx_chunks
                    )
                ),
            }

        return outcomes


def build_expected_manifest(fixture_dir: Path) -> dict[str, Any]:
    """Build the ``expected.json`` payload from the installed packages and current arms."""
    return {
        "versions": get_installed_converter_versions(),
        "arms": {arm_name: evaluate_arm_outcomes(fixture_dir, arm_name) for arm_name in ARM_NAMES},
    }


def generate_fixtures(output_dir: Path = FIXTURES_DIR) -> Path:
    """Write ``two_column.pdf``, ``merged_cells.xlsx``, and ``expected.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PDF_NAME).write_bytes(build_two_column_pdf_bytes())
    (output_dir / XLSX_NAME).write_bytes(build_merged_cells_xlsx_bytes())
    manifest = build_expected_manifest(output_dir)
    expected_path = output_dir / EXPECTED_NAME
    expected_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return expected_path


def main() -> None:
    """CLI entry point for regenerating regression fixtures and ``expected.json``."""
    expected_path = generate_fixtures()
    print(f"Wrote regression fixtures and {expected_path}")


if __name__ == "__main__":
    main()
