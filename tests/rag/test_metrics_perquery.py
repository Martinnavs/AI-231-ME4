"""Offline tests for direct digit diagnostics, --per-query JSONL, and paired comparison."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import convert, metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRED_SCRIPT_PATH = REPO_ROOT / "scripts" / "rag_paired_compare.py"


def _load_paired_module() -> Any:
    spec = importlib.util.spec_from_file_location("rag_paired_compare", PAIRED_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_paired_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _make_xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Budget"
    ws.append(["Site", "Allocation_PHP"])
    ws.append(["Station Alpha", 45000])
    ws.append(["Station Beta", 32500])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Turbine Specs"
    slide.placeholders[1].text = "Rated output is 1250 watts at 240 volts."
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_html_bytes() -> bytes:
    pytest.importorskip("lxml")
    return (
        b"<!doctype html><html><head><title>Valve Guide</title></head>"
        b"<body><h1>Valve Pressure</h1>"
        b"<p>Keep intake pressure between 45 and 60 psi during startup.</p>"
        b"</body></html>"
    )


def test_direct_digit_helper_and_report_json_with_xlsx_pptx_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xlsx_bytes = _make_xlsx_bytes()
    pptx_bytes = _make_pptx_bytes()
    html_bytes = _make_html_bytes()

    for kind, raw in (("xlsx", xlsx_bytes), ("pptx", pptx_bytes), ("html", html_bytes)):
        raw_text, canon_text, dropped_lines, legacy_pages = metrics._convert_raw_and_canonical(
            raw, kind, "direct"
        )
        assert raw_text
        assert canon_text
        assert legacy_pages is None
        if kind in ("xlsx", "pptx"):
            assert dropped_lines

    corpus = tmp_path / "direct_corpus"
    corpus.mkdir()
    (corpus / "budget.xlsx").write_bytes(xlsx_bytes)
    (corpus / "deck.pptx").write_bytes(pptx_bytes)
    (corpus / "valve.html").write_bytes(html_bytes)

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "x1",
                "q": "What is Station Alpha allocation?",
                "type": "spreadsheet_cell",
                "relevant": [{"file": "budget.xlsx", "contains": "Station Alpha"}],
            },
            {
                "id": "p1",
                "q": "What is the turbine rated output?",
                "type": "numeric",
                "relevant": [{"file": "deck.pptx", "contains": "1250 watts"}],
            },
            {
                "id": "h1",
                "q": "What is the intake pressure range?",
                "type": "numeric",
                "relevant": [{"file": "valve.html", "contains": "45 and 60 psi"}],
            },
        ],
    )

    for conv_spec in ("default", "pptx=direct,xlsx=direct,html=direct"):
        report_path = tmp_path / f"report_{conv_spec.replace('=', '_').replace(',', '_')}.json"
        rc = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--converters",
                conv_spec,
                "--match",
                "normalized",
                "--report-json",
                str(report_path),
            ]
        )
        assert rc == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        arm = payload["arms"][0]
        assert arm["digit_mismatches"]["mismatch_count"] == 0
        assert arm["digit_mismatches"]["checked_files"] == 3
        for entry in arm["files"]:
            assert entry["status"] == "ok"
            assert entry["digit_mismatch"] is False
            assert "error" not in (entry["digit_check"] or {})
        assert not (corpus / ".rag_cache").exists()

    # Preflight failure and raw_text-less fallback remain safe.
    bad_raw, bad_canon, bad_dropped, _ = metrics._convert_raw_and_canonical(
        b"not-a-zip", "xlsx", "direct"
    )
    assert (bad_raw, bad_canon, bad_dropped) == ("", "", [])

    monkeypatch.setattr(
        convert,
        "convert_with_direct",
        lambda _raw, _kind: convert.ConvertResult(
            text="<!-- rag:slide=1 -->\nOutput 999 kW.",
            status="ok",
            converter="direct-test",
            raw_text="",
        ),
    )
    fb_raw, fb_canon, fb_dropped, _ = metrics._convert_raw_and_canonical(b"dummy", "pptx", "direct")
    assert fb_raw == "Output 999 kW."
    assert "<!-- rag:slide=1 -->" in fb_canon
    assert fb_dropped == []


def test_per_query_rows_correct_and_excludes_non_answerable(tmp_path: Path) -> None:
    golden_rows: list[dict[str, Any]] = [
        {
            "id": "q-rank1",
            "q": "alpha query",
            "type": "lexical",
            "format": "md",
            "difficulty": ["simple"],
            "relevant": [{"file": "doc1.md", "contains": "alpha anchor"}],
        },
        {
            "id": "q-rank4",
            "q": "beta query",
            "type": "numeric",
            "difficulty": "hard",
            "relevant": [{"file": "doc4.txt", "contains": "beta 42"}],
        },
        {
            "id": "q-rank5",
            "q": "gamma query",
            "type": "identifier",
            "relevant": [{"file": "doc5.md", "contains": "GAMMA-9"}],
        },
        {
            "id": "q-miss",
            "q": "delta query",
            "type": "paraphrase",
            "relevant": [{"file": "doc6.md", "contains": "delta absent"}],
        },
        {
            "id": "q-unans",
            "q": "unanswerable query",
            "type": "unanswerable",
            "relevant": [],
        },
        {
            "id": "q-off",
            "q": "off topic query",
            "type": "off_topic",
            "relevant": [],
        },
    ]

    def _passage(idx: int, src: str, text: str, score: float) -> dict[str, Any]:
        return {"source": src, "id": f"{src}#0", "text": text, "score": score}

    rankings: dict[str, list[dict[str, Any]]] = {
        "alpha query": [
            _passage(1, "doc1.md", "alpha anchor here", 3.12345),
            _passage(2, "d2.md", "noise", 2.0),
            _passage(3, "d3.md", "noise", 1.5),
            _passage(4, "d4.md", "noise", 1.0),
            _passage(5, "d5.md", "noise", 0.5),
            _passage(6, "d6.md", "noise", 0.1),
        ],
        "beta query": [
            _passage(1, "d1.md", "miss", 5.0),
            _passage(2, "d2.md", "miss", 4.0),
            _passage(3, "d3.md", "miss", 3.0),
            _passage(4, "doc4.txt", "contains beta 42 value", 2.5),
            _passage(5, "d5.md", "miss", 1.0),
        ],
        "gamma query": [
            _passage(1, "d1.md", "miss", 5.0),
            _passage(2, "d2.md", "miss", 4.0),
            _passage(3, "d3.md", "miss", 3.0),
            _passage(4, "d4.md", "miss", 2.0),
            _passage(5, "doc5.md", "code GAMMA-9", 1.25),
        ],
        "delta query": [
            _passage(1, "d1.md", "miss", 1.0),
            _passage(2, "d2.md", "miss", 0.9),
        ],
    }

    def fake_search(query: str, k: int) -> list[dict[str, Any]]:
        assert k >= 5
        assert query in rankings
        return rankings[query][:k]

    pq_rows: list[dict[str, Any]] = []
    report = metrics.evaluate(fake_search, golden_rows, ks=(1, 3, 5), per_query_out=pq_rows)
    assert report["overall"]["n"] == 4
    assert [r["id"] for r in pq_rows] == ["q-rank1", "q-rank4", "q-rank5", "q-miss"]

    by_id = {r["id"]: r for r in pq_rows}
    r1 = by_id["q-rank1"]
    assert (r1["hit1"], r1["hit3"], r1["hit4"], r1["hit5"]) == (True, True, True, True)
    assert r1["first_hit_rank"] == 1
    assert r1["format"] == "md"
    assert r1["difficulty"] == ["simple"]
    assert len(r1["top"]) == 5
    assert r1["top"][0] == {"source": "doc1.md", "id": "doc1.md#0", "score": 3.1235}

    r4 = by_id["q-rank4"]
    assert (r4["hit1"], r4["hit3"], r4["hit4"], r4["hit5"]) == (False, False, True, True)
    assert r4["first_hit_rank"] == 4
    assert r4["format"] == "txt"
    assert r4["difficulty"] == ["hard"]

    r5 = by_id["q-rank5"]
    assert (r5["hit1"], r5["hit3"], r5["hit4"], r5["hit5"]) == (False, False, False, True)
    assert r5["first_hit_rank"] == 5

    rm = by_id["q-miss"]
    assert (rm["hit1"], rm["hit3"], rm["hit4"], rm["hit5"]) == (False, False, False, False)
    assert rm["first_hit_rank"] is None
    assert len(rm["top"]) == 2

    # End-to-end CLI test over a tmp folder with --per-query.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for idx in range(1, 7):
        (corpus / f"f{idx}.md").write_text(
            f"# Doc {idx}\n\nUnique marker_token_{idx} value {idx * 100}.\n",
            encoding="utf-8",
        )
    golden_path = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden_path,
        [
            {
                "id": "ans-1",
                "q": "marker_token_1",
                "type": "lexical",
                "relevant": [{"file": "f1.md", "contains": "Unique marker_token_1 value 100."}],
            },
            {
                "id": "unans-1",
                "q": "missing question",
                "type": "unanswerable",
                "relevant": [],
            },
        ],
    )
    pq_out = tmp_path / "out" / "per_query.jsonl"
    rc = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden_path),
            "--per-query",
            str(pq_out),
        ]
    )
    assert rc == 0
    lines = [json.loads(ln) for ln in pq_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    row = lines[0]
    assert set(row) == {
        "id",
        "type",
        "format",
        "difficulty",
        "q",
        "hit1",
        "hit3",
        "hit4",
        "hit5",
        "first_hit_rank",
        "top",
    }
    assert row["id"] == "ans-1"
    assert row["hit1"] is True
    assert row["hit4"] is True
    assert row["first_hit_rank"] == 1
    assert len(row["top"]) == 5
    assert set(row["top"][0]) == {"source", "id", "score"}
    assert not (corpus / ".rag_cache").exists()


def test_per_query_refuses_path_inside_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sub").mkdir()
    (corpus / "a.md").write_text("# A\n\nValve pressure 40 psi.\n", encoding="utf-8")
    golden_path = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden_path,
        [
            {
                "id": "q1",
                "q": "valve pressure",
                "type": "lexical",
                "relevant": [{"file": "a.md", "contains": "Valve pressure 40 psi."}],
            }
        ],
    )

    sym_into_corpus = tmp_path / "sym_into_corpus"
    sym_into_corpus.symlink_to(corpus)

    bad_targets = [
        corpus / "per_query.jsonl",
        corpus / "sub" / "per_query.jsonl",
        corpus,
        corpus / "sub" / ".." / "escaped.jsonl",
        sym_into_corpus / "via_symlink.jsonl",
    ]
    for bad_path in bad_targets:
        rc = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden_path),
                "--per-query",
                str(bad_path),
            ]
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert "must not be inside" in err

    assert sorted(p.name for p in corpus.iterdir()) == ["a.md", "sub"]
    assert list((corpus / "sub").iterdir()) == []

    rc_combo = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden_path),
            "--compare-arms",
            "legacy,default",
            "--per-query",
            str(tmp_path / "outside.jsonl"),
        ]
    )
    assert rc_combo == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_paired_compare_stdlib_only_and_known_mcnemar_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Verify scripts/rag_paired_compare.py imports stdlib modules only.
    tree = ast.parse(PAIRED_SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= sys.stdlib_module_names

    paired = _load_paired_module()

    # Exact McNemar two-sided p-value benchmarks.
    assert paired.mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert paired.mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert paired.mcnemar_exact_p(6, 0) == pytest.approx(0.03125)
    assert paired.mcnemar_exact_p(3, 3) == pytest.approx(1.0)
    assert paired.mcnemar_exact_p(0, 0) == pytest.approx(1.0)

    # Wilson 95% interval sanity check (70/112 from R1).
    lo, hi = paired.wilson_ci(70, 112)
    assert lo == pytest.approx(0.533, abs=1e-3)
    assert hi == pytest.approx(0.709, abs=1e-3)

    def _run_pair(
        a_flags: list[bool],
        b_flags: list[bool],
        *,
        metric: str = "hit4",
        tag: str,
    ) -> str:
        a_rows = [{"id": f"q{i}", metric: a_flags[i]} for i in range(len(a_flags))]
        # Reverse B rows to verify join-by-id regardless of row order.
        b_rows = list(reversed([{"id": f"q{i}", metric: b_flags[i]} for i in range(len(b_flags))]))
        a_path = tmp_path / f"{tag}_a.jsonl"
        b_path = tmp_path / f"{tag}_b.jsonl"
        _write_jsonl(a_path, a_rows)
        _write_jsonl(b_path, b_rows)
        rc = paired.main([str(a_path), str(b_path), "--metric", metric])
        assert rc == 0
        return capsys.readouterr().out

    # Case 1: b=5, c=0 -> p=0.0625, verdict "inside noise"
    out_5_0 = _run_pair(
        [False, False, False, False, False, True, True],
        [True, True, True, True, True, True, True],
        metric="hit4",
        tag="case_5_0",
    )
    assert "n: 7" in out_5_0
    assert "b=5, c=0" in out_5_0
    assert "0.062500" in out_5_0
    assert "inside noise" in out_5_0
    assert "q0, q1, q2, q3, q4" in out_5_0

    # Case 2: b=6, c=0 -> p=0.03125 (significant, still exits 0)
    out_6_0 = _run_pair(
        [False] * 6 + [True] * 2,
        [True] * 8,
        metric="hit3",
        tag="case_6_0",
    )
    assert "metric: hit3  n: 8" in out_6_0
    assert "b=6, c=0" in out_6_0
    assert "0.031250" in out_6_0
    assert "inside noise" not in out_6_0

    # Case 3: b=3, c=3 -> p=1.0, both flip lists printed, verdict "inside noise"
    out_3_3 = _run_pair(
        [False, False, False, True, True, True, True],
        [True, True, True, False, False, False, True],
        metric="hit5",
        tag="case_3_3",
    )
    assert "metric: hit5  n: 7" in out_3_3
    assert "b=3, c=3" in out_3_3
    assert "1.000000" in out_3_3
    assert "flipped A=0 -> B=1 (3): q0, q1, q2" in out_3_3
    assert "flipped A=1 -> B=0 (3): q3, q4, q5" in out_3_3
    assert "inside noise" in out_3_3

    # Also verify standalone execution via subprocess with bare python3.
    proc = subprocess.run(
        [
            sys.executable,
            str(PAIRED_SCRIPT_PATH),
            str(tmp_path / "case_5_0_a.jsonl"),
            str(tmp_path / "case_5_0_b.jsonl"),
            "--metric",
            "hit4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "inside noise" in proc.stdout
    assert "0.062500" in proc.stdout
