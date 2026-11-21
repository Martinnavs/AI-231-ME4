"""Offline tests for external folder evaluation, --compare-arms, --match, and --report-json."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_loaders_binary import _build_docx_bytes, _build_pdf_bytes, _docx_p, _wrap_document_xml

from voice_agent.rag import metrics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _folder_snapshot(folder: Path) -> dict[str, tuple[int, str]]:
    """Capture relative path -> (mtime_ns, sha256) for every entry under ``folder``."""
    snap: dict[str, tuple[int, str]] = {}
    for entry in sorted(folder.rglob("*")):
        rel = entry.relative_to(folder).as_posix()
        st = entry.stat()
        digest = hashlib.sha256(entry.read_bytes()).hexdigest() if entry.is_file() else "<dir>"
        snap[rel] = (st.st_mtime_ns, digest)
    return snap


def _make_xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Revenue"
    ws.append(["Region", "Annual_Budget", "Status"])
    ws.append(["North Zone", 12500, "Approved"])
    ws.append(["South Zone", 8400, "Pending"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Operations Table"
    slide.placeholders[1].text = "Quarterly pump inventory."
    table_shape = slide.shapes.add_table(
        2,
        2,
        pptx.util.Inches(1),
        pptx.util.Inches(3),
        pptx.util.Inches(5),
        pptx.util.Inches(1.5),
    )
    table = table_shape.table
    table.cell(0, 0).text = "Unit_Name"
    table.cell(0, 1).text = "Capacity_Liters"
    table.cell(1, 0).text = "Pump Alpha"
    table.cell(1, 1).text = "2500"
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_normalize_for_match_and_is_relevant_modes() -> None:
    assert metrics.normalize_for_match("Revenue: $1,234,567.50!") == "revenue 1234567 50"
    assert metrics.normalize_for_match("Unit_Name — O’Brien’s") == "unit name o'brien's"
    assert metrics.normalize_for_match("100, 200") == "100 200"

    passage = {
        "source": "budget.xlsx",
        "text": "Region: North Zone; Annual_Budget: 12,500; Status: Approved.",
    }
    relevant = [{"file": "budget.xlsx", "contains": "region north zone annual budget 12500"}]

    assert not metrics.is_relevant(passage, relevant, match="exact")
    assert metrics.is_relevant(passage, relevant, match="normalized")

    with pytest.raises(ValueError, match="unsupported match mode"):
        metrics.is_relevant(passage, relevant, match="fuzzy")


def test_evaluate_supports_table_cell_and_spreadsheet_cell_and_match_modes() -> None:
    golden = [
        {
            "id": "tc-1",
            "q": "What is the pump capacity?",
            "type": "table_cell",
            "relevant": [{"file": "deck.pptx", "contains": "Pump Alpha; Capacity Liters: 2,500"}],
        },
        {
            "id": "sc-1",
            "q": "What is North Zone annual budget?",
            "type": "spreadsheet_cell",
            "relevant": [{"file": "sheet.xlsx", "contains": "North Zone; Annual Budget: 12,500"}],
        },
        {
            "id": "ua-1",
            "q": "Missing fact",
            "type": "unanswerable",
            "relevant": [],
        },
    ]

    rankings = {
        "What is the pump capacity?": [
            {"source": "deck.pptx", "text": "Unit_Name: Pump Alpha; Capacity_Liters: 2500"}
        ],
        "What is North Zone annual budget?": [
            {"source": "sheet.xlsx", "text": "Region: North Zone; Annual_Budget: 12500"}
        ],
    }

    def fake_search(query: str, k: int) -> list[dict[str, Any]]:
        assert query in rankings
        return rankings[query][:k]

    exact_report = metrics.evaluate(fake_search, golden, ks=(1, 3), match="exact")
    assert exact_report["overall"]["n"] == 2
    assert exact_report["overall"]["recall"][1] == pytest.approx(0.0)
    assert set(exact_report["by_type"]) == {"table_cell", "spreadsheet_cell"}

    norm_report = metrics.evaluate(fake_search, golden, ks=(1, 3), match="normalized")
    assert norm_report["overall"]["n"] == 2
    assert norm_report["overall"]["recall"][1] == pytest.approx(1.0)
    assert norm_report["overall"]["mrr"] == pytest.approx(1.0)
    assert norm_report["by_type"]["table_cell"]["recall"][1] == pytest.approx(1.0)
    assert norm_report["by_type"]["spreadsheet_cell"]["recall"][1] == pytest.approx(1.0)


def test_dump_passages_refused_without_explicit_path_for_non_default_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "ext_corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text("# Notes\n\nBattery lasts 18 hours.\n", encoding="utf-8")
    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "q1",
                "q": "How long does the battery last?",
                "type": "lexical",
                "relevant": [{"file": "notes.md", "contains": "Battery lasts 18 hours."}],
            }
        ],
    )

    default_snap = metrics.DEFAULT_PASSAGES_PATH
    snap_before = default_snap.read_bytes() if default_snap.is_file() else None

    rc_refused = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--dump-passages",
        ]
    )
    err_out = capsys.readouterr().err
    assert rc_refused == 2
    assert "explicit PATH" in err_out
    if snap_before is not None:
        assert default_snap.read_bytes() == snap_before

    explicit_out = tmp_path / "out" / "custom_passages.json"
    rc_ok = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--dump-passages",
            str(explicit_out),
        ]
    )
    assert rc_ok == 0
    assert explicit_out.is_file()
    parsed = json.loads(explicit_out.read_text(encoding="utf-8"))
    assert parsed["k"] == 4
    assert "How long does the battery last?" in parsed["questions"]
    assert not (corpus / ".rag_cache").exists()


def test_compare_arms_subprocesses_diagnostics_and_readonly_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "external_docs"
    corpus.mkdir()

    (corpus / "guide.md").write_text(
        "# Water Pump Guide\n\nPrime the intake line for 30 seconds before startup.\n",
        encoding="utf-8",
    )
    (corpus / "schedule.csv").write_text(
        "shift,lead,budget_php\nmorning,Elena,12500\n",
        encoding="utf-8",
    )
    (corpus / "empty.txt").write_bytes(b"")
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4\nnot-a-valid-pdf")
    (corpus / "memo.docx").write_bytes(
        _build_docx_bytes(_wrap_document_xml(_docx_p("Backup generator inspection every Friday.")))
    )
    (corpus / "manual.pdf").write_bytes(
        _build_pdf_bytes(
            [
                "Page one mentions cooling tower valve alpha",
                "continues across boundary to page two.",
            ]
        )
    )

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "lex-1",
                "q": "How long should the intake line be primed?",
                "type": "lexical",
                "relevant": [
                    {
                        "file": "guide.md",
                        "contains": "Prime the intake line for 30 seconds before startup.",
                    }
                ],
            },
            {
                "id": "tc-1",
                "q": "What is the morning shift budget?",
                "type": "table_cell",
                "relevant": [
                    {
                        "file": "schedule.csv",
                        "contains": "shift morning lead elena budget php 12,500",
                    }
                ],
            },
            {
                "id": "split-1",
                "q": "What does cooling tower valve alpha do?",
                "type": "lexical",
                "relevant": [
                    {
                        "file": "manual.pdf",
                        "contains": "cooling tower valve alpha continues across boundary",
                    }
                ],
            },
        ],
    )

    report_path = tmp_path / "reports" / "compare.json"
    before_snap = _folder_snapshot(corpus)

    os.chmod(corpus, 0o555)
    try:
        rc = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--compare-arms",
                "legacy,pypdf",
                "--match",
                "normalized",
                "--report-json",
                str(report_path),
            ]
        )
    finally:
        os.chmod(corpus, 0o755)

    assert rc == 0
    captured = capsys.readouterr()
    assert "=== arm: legacy " in captured.out
    assert "=== arm: pypdf " in captured.out
    assert "table_cell" in captured.out
    assert "peak_rss=" in captured.out
    assert "cross_page_splits:" in captured.out
    assert "digit_mismatches:" in captured.out
    assert "bad.pdf: status=unreadable" in captured.out
    assert "empty.txt: status=empty" in captured.out

    assert not (corpus / ".rag_cache").exists()
    assert _folder_snapshot(corpus) == before_snap

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["match"] == "normalized"
    assert [arm["arm"] for arm in payload["arms"]] == ["legacy", "pypdf"]

    for arm in payload["arms"]:
        assert arm["peak_rss_bytes"] > 0
        assert arm["peak_rss_mb"] > 0
        assert arm["timings"]["refresh_s"] >= 0.0
        assert arm["timings"]["conversion_s"] >= 0.0
        assert "table_cell" in arm["metrics"]["by_type"]
        assert arm["metrics"]["by_type"]["table_cell"]["recall"]["1"] == pytest.approx(1.0)

        statuses = {f["name"]: f["status"] for f in arm["files"]}
        assert statuses["guide.md"] == "ok"
        assert statuses["schedule.csv"] == "ok"
        assert statuses["memo.docx"] == "ok"
        assert statuses["manual.pdf"] == "ok"
        assert statuses["empty.txt"] == "empty"
        assert statuses["bad.pdf"] == "unreadable"

        assert arm["fallback_rates"]["ok_file_count"] == 4
        assert arm["fallback_rates"]["total_file_count"] == 6
        assert arm["digit_mismatches"]["mismatch_count"] == 0
        assert arm["cross_page_splits"]["count"] == 1
        assert arm["cross_page_splits"]["pages_ending_mid_sentence"] == 1
        assert arm["cross_page_splits"]["details"][0]["id"] == "split-1"


def test_compare_arms_with_markitdown_office_tables_and_no_folder_writes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("markitdown")

    corpus = tmp_path / "office_corpus"
    corpus.mkdir()
    (corpus / "budget.xlsx").write_bytes(_make_xlsx_bytes())
    (corpus / "deck.pptx").write_bytes(_make_pptx_bytes())
    (corpus / "report.pdf").write_bytes(
        _build_pdf_bytes(["Quarterly review 2026 summary.\nTotal units 420."])
    )

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "sc-1",
                "q": "What is the North Zone annual budget?",
                "type": "spreadsheet_cell",
                "relevant": [
                    {
                        "file": "budget.xlsx",
                        "contains": "North Zone; Annual Budget: 12,500",
                    }
                ],
            },
            {
                "id": "tc-1",
                "q": "What is Pump Alpha capacity?",
                "type": "table_cell",
                "relevant": [
                    {
                        "file": "deck.pptx",
                        "contains": "Pump Alpha; Capacity Liters: 2,500",
                    }
                ],
            },
            {
                "id": "num-1",
                "q": "How many total units in the quarterly review?",
                "type": "numeric",
                "relevant": [{"file": "report.pdf", "contains": "Total units 420."}],
            },
        ],
    )

    report_path = tmp_path / "markitdown_report.json"
    before_snap = _folder_snapshot(corpus)

    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--compare-arms",
            "legacy,markitdown",
            "--match",
            "normalized",
            "--report-json",
            str(report_path),
        ]
    )
    assert rc == 0
    assert not (corpus / ".rag_cache").exists()
    assert _folder_snapshot(corpus) == before_snap

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    by_arm = {arm["arm"]: arm for arm in payload["arms"]}
    md_arm = by_arm["markitdown"]
    assert md_arm["peak_rss_bytes"] > 0
    assert md_arm["metrics"]["by_type"]["spreadsheet_cell"]["recall"]["1"] == pytest.approx(1.0)
    assert md_arm["metrics"]["by_type"]["table_cell"]["recall"]["1"] == pytest.approx(1.0)
    assert md_arm["metrics"]["by_type"]["numeric"]["recall"]["1"] == pytest.approx(1.0)
    assert md_arm["digit_mismatches"]["mismatch_count"] == 0


def test_single_arm_report_json_and_cli_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "single"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Alpha turbine output is 1,500 MW.\n", encoding="utf-8")
    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "num-1",
                "q": "What is the alpha turbine output?",
                "type": "numeric",
                "relevant": [{"file": "a.txt", "contains": "output is 1500 mw"}],
            }
        ],
    )

    report_path = tmp_path / "single_report.json"
    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--converters",
            "pypdf",
            "--match",
            "normalized",
            "--report-json",
            str(report_path),
        ]
    )
    assert rc == 0
    assert not (corpus / ".rag_cache").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(payload["arms"]) == 1
    assert payload["arms"][0]["metrics"]["overall"]["recall"]["1"] == pytest.approx(1.0)

    for bad_arms in ("", "legacy,", "legacy,nonexistent_arm"):
        rc_bad = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--compare-arms",
                bad_arms,
            ]
        )
        assert rc_bad == 2
        assert "error:" in capsys.readouterr().err

    rc_combo = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--compare-arms",
            "legacy,pypdf",
            "--dump-passages",
            str(tmp_path / "snap.json"),
        ]
    )
    assert rc_combo == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_evaluate_arm_reports_fallbacks_and_digit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "fb_corpus"
    corpus.mkdir()
    (corpus / "doc.pdf").write_bytes(
        _build_pdf_bytes(["Reading 1234 volts measured at the sensor."])
    )
    (corpus / "note.txt").write_text("Reading 5678 amps.\n", encoding="utf-8")

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "n1",
                "q": "What is the voltage reading?",
                "type": "numeric",
                "relevant": [{"file": "doc.pdf", "contains": "1234 volts"}],
            }
        ],
    )

    real_docindex_cls = metrics.DocIndex

    class _FallbackDocIndex(real_docindex_cls):  # type: ignore[misc, valid-type]
        def refresh(self) -> Any:
            rep = super().refresh()
            self._chunks = [
                type(c)(
                    **{
                        **c.__dict__,
                        "converter": (
                            "pypdf-7.1.0+pp1/fallback=pypdf-pages"
                            if c.source == "doc.pdf"
                            else "legacy/fallback=legacy"
                        ),
                    }
                )
                for c in self._chunks
            ]
            return rep

    monkeypatch.setattr(metrics, "DocIndex", _FallbackDocIndex)
    monkeypatch.setattr(
        metrics,
        "_convert_raw_and_canonical",
        lambda raw_bytes, kind, mode: (
            ("Reading 1234 volts.", "Reading 999 volts.", [], None)
            if kind == "pdf"
            else ("Reading 5678 amps.\n", "Reading 5678 amps.\n", [], None)
        ),
    )

    report = metrics.evaluate_arm(
        "pypdf",
        folder=corpus,
        golden_path=golden,
    )
    assert report["fallback_rates"]["pdf_pypdf_fallback_count"] == 1
    assert report["fallback_rates"]["pdf_pypdf_fallback_rate"] == pytest.approx(1.0)
    assert report["fallback_rates"]["legacy_fallback_count"] == 1
    assert report["fallback_rates"]["legacy_fallback_rate"] == pytest.approx(0.5)
    assert report["digit_mismatches"]["mismatch_count"] == 1
    assert report["digit_mismatches"]["mismatched_files"] == ["doc.pdf"]

    pdf_entry = next(entry for entry in report["files"] if entry["name"] == "doc.pdf")
    assert pdf_entry["digit_mismatch"] is True
    assert pdf_entry["digit_check"]["missing"] == ["1234"]
    assert pdf_entry["digit_check"]["extra"] == ["999"]
    formatted = metrics.format_arm_report(report)
    assert "digit_mismatch=true" in formatted
    assert "fallback=pypdf-pages" in formatted


def test_dump_passages_ignores_symlinked_files_and_dirs(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.md"
    outside_file.write_text(
        "# Secret\n\nfile_query_alpha OUTSIDE_FILE_MARKER_XYZ999.\n",
        encoding="utf-8",
    )
    outside_sub = outside_dir / "sub"
    outside_sub.mkdir()
    (outside_sub / "nested.md").write_text(
        "# Nested\n\ndir_query_beta OUTSIDE_DIR_MARKER_ABC888.\n",
        encoding="utf-8",
    )
    outside_mtime_before = outside_file.stat().st_mtime_ns

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text("# Notes\n\nBattery capacity is 18 hours.\n", encoding="utf-8")
    (corpus / "leak_file.md").symlink_to(outside_file)
    (corpus / "leak_dir").symlink_to(outside_sub)

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "q1",
                "q": "file_query_alpha",
                "type": "unanswerable",
                "relevant": [],
            },
            {
                "id": "q2",
                "q": "dir_query_beta",
                "type": "unanswerable",
                "relevant": [],
            },
            {
                "id": "q3",
                "q": "How long is battery capacity?",
                "type": "lexical",
                "relevant": [{"file": "notes.md", "contains": "Battery capacity is 18 hours."}],
            },
        ],
    )

    out_path = tmp_path / "out" / "passages.json"
    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--dump-passages",
            str(out_path),
        ]
    )
    assert rc == 0
    raw_dump = out_path.read_text(encoding="utf-8")
    assert "OUTSIDE_FILE_MARKER_XYZ999" not in raw_dump
    assert "OUTSIDE_DIR_MARKER_ABC888" not in raw_dump
    parsed = json.loads(raw_dump)
    assert parsed["questions"]["file_query_alpha"] == []
    assert parsed["questions"]["dir_query_beta"] == []
    assert parsed["questions"]["How long is battery capacity?"][0]["source"] == "notes.md"
    assert outside_file.stat().st_mtime_ns == outside_mtime_before
