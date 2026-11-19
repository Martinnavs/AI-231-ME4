"""Offline tests for office fixtures (xlsx/pptx) and office golden rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import openpyxl  # type: ignore[import-untyped]
import pptx  # type: ignore[import-untyped]
import pytest

from voice_agent.rag.docindex import DocIndex
from voice_agent.rag.metrics import (
    evaluate,
    format_metrics_table,
    load_golden,
    normalize_whitespace,
)

_GEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "make_rag_office_fixtures.py"
_SPEC = importlib.util.spec_from_file_location("make_rag_office_fixtures", _GEN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GEN)

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICE_DIR = _GEN.FIXTURES_DIR
GOLDEN_OFFICE_PATH = REPO_ROOT / "config" / "eval" / "rag_golden_office.jsonl"
MARKITDOWN_OFFICE_CONVERTERS = {"xlsx": "markitdown", "pptx": "markitdown"}
DIRECT_OFFICE_CONVERTERS = {"pptx": "direct", "xlsx": "direct", "html": "direct", "epub": "direct"}
EXPECTED_OFFICE_TYPE_COUNTS = {"spreadsheet_cell": 3, "table_cell": 3}

# Measured R@3 floor on config/eval/rag_golden_office.jsonl when 06c landed.
OFFICE_R_AT_3_FLOOR = 1.0
# Measured R@3 floor under the direct arm (all anchors are converter-independent).
OFFICE_DIRECT_R_AT_3_FLOOR = 1.0


def test_legacy_arm_reports_office_files_unsupported() -> None:
    index = DocIndex(OFFICE_DIR, converters={}, canonical_cache_dir=None)
    index.refresh()

    assert index.chunks == ()
    records = {rec.name: rec for rec in index.records()}
    assert set(records) == {_GEN.XLSX_NAME, _GEN.PPTX_NAME}
    assert records[_GEN.XLSX_NAME].status == "unsupported"
    assert records[_GEN.PPTX_NAME].status == "unsupported"
    failures = {name: status for name, status, _ in index.failures()}
    assert failures == {_GEN.XLSX_NAME: "unsupported", _GEN.PPTX_NAME: "unsupported"}


def test_office_fixtures_structure_and_generator(tmp_path: Path) -> None:
    xlsx_path = OFFICE_DIR / _GEN.XLSX_NAME
    pptx_path = OFFICE_DIR / _GEN.PPTX_NAME
    assert xlsx_path.is_file()
    assert pptx_path.is_file()

    wb = openpyxl.load_workbook(xlsx_path)
    assert wb.sheetnames == ["Operations", "Hardware"]
    assert {str(rng) for rng in wb["Operations"].merged_cells.ranges} == {"A1:D1"}

    prs = pptx.Presentation(str(pptx_path))
    assert len(prs.slides) == 2
    assert any(shape.has_table for shape in prs.slides[0].shapes)
    assert not any(shape.has_table for shape in prs.slides[1].shapes)

    gen_xlsx, gen_pptx = _GEN.generate_fixtures(tmp_path)
    assert gen_xlsx.is_file() and gen_xlsx.stat().st_size > 0
    assert gen_pptx.is_file() and gen_pptx.stat().st_size > 0


def test_office_golden_schema_and_verbatim_anchors() -> None:
    pytest.importorskip("markitdown")
    golden = load_golden(GOLDEN_OFFICE_PATH)
    assert len(golden) == 6

    index = DocIndex(
        OFFICE_DIR,
        converters=MARKITDOWN_OFFICE_CONVERTERS,
        canonical_cache_dir=None,
    )
    index.refresh()
    assert index.failures() == []
    assert len(index.records()) == 2

    chunks_by_source: dict[str, list[str]] = {}
    for chunk in index.chunks:
        chunks_by_source.setdefault(chunk.source, []).append(normalize_whitespace(chunk.text))

    counts: dict[str, int] = {}
    ids: list[str] = []
    for row in golden:
        row_id = str(row["id"])
        q_type = str(row["type"])
        relevant = row["relevant"]
        ids.append(row_id)
        counts[q_type] = counts.get(q_type, 0) + 1

        assert relevant, f"{row_id} must have at least one relevant anchor"
        for rel in relevant:
            file_name = str(rel["file"])
            contains = str(rel["contains"])
            assert len(contains) < 150, f"{row_id} anchor exceeds 150 chars"
            norm_contains = normalize_whitespace(contains)
            file_chunks = chunks_by_source.get(file_name, [])
            assert any(norm_contains in text for text in file_chunks), (
                f"{row_id} anchor not found intact in any chunk of {file_name}"
            )

    assert len(ids) == len(set(ids))
    assert counts == EXPECTED_OFFICE_TYPE_COUNTS


def test_office_golden_retrieval_floor() -> None:
    pytest.importorskip("markitdown")
    golden = load_golden(GOLDEN_OFFICE_PATH)
    index = DocIndex(
        OFFICE_DIR,
        converters=MARKITDOWN_OFFICE_CONVERTERS,
        canonical_cache_dir=None,
    )
    index.refresh()
    assert index.failures() == []

    report = evaluate(index.search, golden, ks=(1, 3, 5))
    print("\n" + format_metrics_table(report, ks=(1, 3, 5)))

    overall = report["overall"]
    assert overall["n"] == 6
    assert overall["recall"][3] >= OFFICE_R_AT_3_FLOOR, f"Office R@3 floor missed: {overall}"
    for q_type in EXPECTED_OFFICE_TYPE_COUNTS:
        assert q_type in report["by_type"]
        assert report["by_type"][q_type]["recall"][3] >= OFFICE_R_AT_3_FLOOR


def test_office_golden_retrieval_floor_direct() -> None:
    golden = load_golden(GOLDEN_OFFICE_PATH)
    index = DocIndex(
        OFFICE_DIR,
        converters=DIRECT_OFFICE_CONVERTERS,
        canonical_cache_dir=None,
    )
    index.refresh()
    assert index.failures() == []
    assert {rec.name: rec.status for rec in index.records()} == {
        _GEN.XLSX_NAME: "ok",
        _GEN.PPTX_NAME: "ok",
    }

    report = evaluate(index.search, golden, ks=(1, 3, 5))
    print("\n" + format_metrics_table(report, ks=(1, 3, 5)))

    overall = report["overall"]
    assert overall["n"] == 6
    assert overall["recall"][3] >= OFFICE_DIRECT_R_AT_3_FLOOR, f"Direct R@3 floor missed: {overall}"
