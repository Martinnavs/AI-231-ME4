"""Parser regression fixture tests across converter arms and version manifest."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import openpyxl  # type: ignore[import-untyped]
import pytest

_GEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "make_rag_regress_fixtures.py"
_SPEC = importlib.util.spec_from_file_location("make_rag_regress_fixtures", _GEN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GEN)

FIXTURES_DIR = _GEN.FIXTURES_DIR
EXPECTED_PATH = FIXTURES_DIR / _GEN.EXPECTED_NAME


def _load_expected() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    return data


def test_converter_versions_match_manifest() -> None:
    manifest = _load_expected()
    versions = manifest["versions"]
    assert set(versions) == set(_GEN.CONVERTER_PACKAGES)
    _GEN.verify_converter_versions(versions)


@pytest.mark.parametrize("package_name", _GEN.CONVERTER_PACKAGES)
def test_converter_version_mismatch_fails_with_exact_message(
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
) -> None:
    manifest = _load_expected()
    real_version = importlib.metadata.version

    def _fake_version(name: str) -> str:
        if name == package_name:
            return "0.0.0-mismatch"
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", _fake_version)

    with pytest.raises(
        AssertionError,
        match=r"^converter version changed: rerun and review expected\.json$",
    ):
        _GEN.verify_converter_versions(manifest["versions"])


def test_regression_outcomes_match_manifest_per_arm() -> None:
    manifest = _load_expected()
    expected_arms = manifest["arms"]
    assert set(expected_arms) == set(_GEN.ARM_NAMES)
    assert "legacy" in expected_arms

    actual_arms = {
        arm_name: _GEN.evaluate_arm_outcomes(FIXTURES_DIR, arm_name) for arm_name in _GEN.ARM_NAMES
    }
    assert actual_arms == expected_arms

    # Key cross-arm regression invariants recorded in expected.json.
    assert actual_arms["legacy"][_GEN.PDF_NAME]["sentence_intact"] is False
    assert actual_arms["pypdf"][_GEN.PDF_NAME]["sentence_intact"] is False
    assert actual_arms["markitdown"][_GEN.PDF_NAME]["sentence_intact"] is True

    assert actual_arms["legacy"][_GEN.XLSX_NAME]["status"] == "unsupported"
    assert actual_arms["legacy"][_GEN.XLSX_NAME]["sheet_locator"] is False
    assert set(actual_arms["direct"]) == {_GEN.XLSX_NAME}
    for canonical_arm in ("pypdf", "markitdown", "direct"):
        xlsx_res = actual_arms[canonical_arm][_GEN.XLSX_NAME]
        assert xlsx_res["status"] == "ok"
        assert xlsx_res["sheet_locator"] is True
        assert xlsx_res["merged_value_present_once"]["vertical_label"] is True
        assert "15" in xlsx_res["digit_runs"]
        assert "85" in xlsx_res["digit_runs"]


def test_generator_reproduces_committed_fixtures(tmp_path: Path) -> None:
    assert _GEN.build_two_column_pdf_bytes() == (FIXTURES_DIR / _GEN.PDF_NAME).read_bytes()

    wb = openpyxl.load_workbook(FIXTURES_DIR / _GEN.XLSX_NAME)
    ws = wb[_GEN.SHEET_NAME]
    merged_ranges = {str(rng) for rng in ws.merged_cells.ranges}
    assert "A1:C1" in merged_ranges
    assert "A3:A4" in merged_ranges
    assert ws["C3"].number_format == "0%"
    assert ws["C4"].number_format == "0%"

    generated_expected = _GEN.generate_fixtures(tmp_path)
    assert generated_expected.read_text(encoding="utf-8") == EXPECTED_PATH.read_text(
        encoding="utf-8"
    )
