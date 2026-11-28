"""scripts/bench_mmlongdoc_extraction.py: the MMLongBench-Doc text-extraction
check (ticket 275).

All through fakes (no relay, no corpus DB): a synthetic sqlite corpus DB and a
generated one-page PDF drive the normalisation (Indian grouping, $, %,
fiscal/FY, the "and" word), the gold-in-text matcher (string / number /
every-List-item), the five-class assignment, the per-PDF health flags, the
table-integrity chunk-locality check, the slide-deck verdict, and the
independent extractors themselves (whatever of pypdf / pdfminer.six /
pypdfium2 imports in the venv).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "bench_mmlongdoc", _REPO / "scripts" / "bench_mmlongdoc.py"
)
assert _SPEC and _SPEC.loader
bm = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_mmlongdoc"] = bm  # so the extraction module's `import` resolves
_SPEC.loader.exec_module(bm)

_SPEC2 = importlib.util.spec_from_file_location(
    "bench_mmlongdoc_extraction", _REPO / "scripts" / "bench_mmlongdoc_extraction.py"
)
assert _SPEC2 and _SPEC2.loader
be = importlib.util.module_from_spec(_SPEC2)
sys.modules["bench_mmlongdoc_extraction"] = be
_SPEC2.loader.exec_module(be)

DOC = "docA.pdf"


# --- fixtures -------------------------------------------------------------------
def make_one_page_pdf(path: Path, text: str) -> None:
    """A minimal one-page PDF with one Helvetica line and a correct xref, so
    every extractor (pypdf, pdfminer.six, pdfium) can read it. ASCII text
    only; no parentheses."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(body)
        out.write(b"\nendobj\n")
    xref_pos = out.tell()
    out.write(b"xref\n0 6\n")
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())
    path.write_bytes(out.getvalue())


def fake_db(path: Path) -> None:
    """A synthetic corpus DB: a healthy doc (2 pages, 50 chars), a zero-chunk
    doc and a thin 1-page doc (500 chars)."""
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE files (rel_key TEXT, status TEXT, n_chunks INTEGER, note TEXT)")
    con.execute(
        "CREATE TABLE chunk_data "
        "(chunk_id TEXT, rel_key TEXT, ord INTEGER, page INTEGER, text TEXT)"
    )
    con.execute("INSERT INTO files VALUES ('healthy.pdf', 'ok', 1, '')")
    con.execute(
        "INSERT INTO files VALUES ('slides.pdf', 'ok', 0, "
        "'scanned: 1 pages without text; 1 images waiting for OCR')"
    )
    con.execute("INSERT INTO files VALUES ('thin.pdf', 'ok', 1, '')")
    con.execute(
        "INSERT INTO chunk_data VALUES "
        "('h1', 'healthy.pdf', 0, 1, 'the total is 1,234.5 million rupees')"
    )
    con.execute("INSERT INTO chunk_data VALUES ('t1', 'thin.pdf', 0, 1, ?)", ("x" * 500,))
    con.commit()
    con.close()


def fake_manifest() -> dict[str, Any]:
    return {
        "docs": [
            {"doc_id": "healthy.pdf", "pages": 2},
            {"doc_id": "slides.pdf", "pages": 1},
            {"doc_id": "thin.pdf", "pages": 1},
        ],
        "scanned_pages": {
            "per_pdf": [
                {"path": "healthy.pdf", "pages": 2, "scanned_pages": 0},
                {"path": "slides.pdf", "pages": 1, "scanned_pages": 1},
                {"path": "thin.pdf", "pages": 1, "scanned_pages": 0},
            ]
        },
    }


def fakes(gold_in_db: str = "1234.5", indep_found: bool = True) -> dict:
    """A minimal ``classify_extraction``-shaped signal set: the doc has one
    chunk (gold_in_db) and one independent extractor (found = indep_found)."""
    return {
        "n_chunks": 1,
        "full_text": gold_in_db,
        "page_texts": [gold_in_db],
        "independent": {"fake": {"found": indep_found, "how": "x"}},
    }


# --- normalisation ---------------------------------------------------------------
def test_norm_gold_indian_grouping_and_symbols() -> None:
    assert be.norm_gold("3,02,16,492") == "30216492"
    assert be.norm_gold("$1,234.50") == "1234.50"
    assert be.norm_gold("12.5%") == "12.5"
    assert be.norm_gold("INR 1,23,45,678.90") == "inr12345678.90"


def test_norm_gold_fiscal_forms_and_case() -> None:
    forms = (
        be.norm_gold("Fiscal Year 2021"),
        be.norm_gold("fiscal 2021"),
        be.norm_gold("FY 2021"),
        be.norm_gold("FY2021"),
        be.norm_gold("FISCALYEAR2021"),
    )
    assert forms == ("fy2021",) * 5


def test_norm_gold_drops_the_word_and_only() -> None:
    assert be.norm_gold("A and B") == "ab"
    assert be.norm_gold("Band") == "band"
    # whitespace is packed too: the comma-glued dataset items become
    # "foodsafetynutritiondistribution", matching spaced table text
    assert be.norm_gold("Food Safety,Nutrition,and Distribution") == (
        "foodsafetynutritiondistribution"
    )
    assert be.norm_gold("Food Safety, Nutrition, and Distribution") == (
        "foodsafetynutritiondistribution"
    )


def test_gold_in_text_number_equivalence() -> None:
    # Indian-grouped text vs a plain gold float: the number matches
    found, how = be.gold_in_text("30216492.00", "Float", "total dividend 3,02,16,492 rupees")
    assert found and how == "number"
    # a different number (or a substring of one) does not match
    found, _ = be.gold_in_text("3", "Int", "twenty three and a 23 percent rise")
    assert not found
    found, how = be.gold_in_text("10,314", "Float", "long-term debt of 10314 million")
    assert found and how == "number"


def test_gold_in_text_string() -> None:
    found, how = be.gold_in_text("Radley Horton", "Str", "led by RADLEY  horton, yes")
    assert found and how == "string"
    found, _ = be.gold_in_text("Radley Horton", "Str", "led by someone else")
    assert not found


def test_gold_in_text_list_requires_every_item() -> None:
    gold = "['alpha', 'beta', 'gamma']"
    found, how = be.gold_in_text(gold, "List", "the items are alpha, beta and gamma")
    assert found and how == "list 3/3"
    # two of three is NOT enough (the ticket-274 matcher used a majority)
    found, _ = be.gold_in_text(gold, "List", "only alpha and beta here")
    assert not found
    # the comma-glued dataset items match spaced table text
    gold2 = "['Food Safety,Nutrition,and Distribution', 'Mental Health and Well-Being']"
    found, how = be.gold_in_text(
        gold2,
        "List",
        "chapters: Food Safety, Nutrition, and Distribution; Mental Health and Well-Being",
    )
    assert found and how == "list 2/2"


def test_classify_extraction_classes() -> None:
    gold, fmt = "1234.5", "Float"
    base = fakes()
    out = be.classify_extraction(gold=gold, answer_format=fmt, **base)
    assert out["class"] == "FOUND-BOTH"
    # the DB loses it: an independent extractor finds it
    out = be.classify_extraction(
        gold=gold,
        answer_format=fmt,
        **{**base, "full_text": "no number here", "page_texts": ["no number here"]},
    )
    assert out["class"] == "FOUND-INDEPENDENT-ONLY"
    assert out["found_db_any"] is False
    assert out["found_independent"] is True
    # the independent extractor sees nothing (e.g. a scanned gold page)
    out = be.classify_extraction(
        gold=gold,
        answer_format=fmt,
        **{**base, "independent": {"fake": {"found": False, "how": ""}}},
    )
    assert out["class"] == "FOUND-DB-ONLY"
    # nowhere at all
    out = be.classify_extraction(
        gold="999.9",
        answer_format="Float",
        **{**base, "independent": {"fake": {"found": False, "how": ""}}},
    )
    assert out["class"] == "ABSENT-BOTH"
    # the gold document has zero chunks in the DB, whatever the extractors say
    out = be.classify_extraction(gold=gold, answer_format=fmt, **{**base, "n_chunks": 0})
    assert out["class"] == "DOC-EMPTY"
    # the on-evidence-pages check is reported alongside the DB-any check
    out = be.classify_extraction(gold=gold, answer_format=fmt, **{**base, "page_texts": []})
    assert out["class"] == "FOUND-BOTH"
    assert out["found_db_pages"] is False


def test_assign_class_priority() -> None:
    assert be.assign_class(0, True, True) == "DOC-EMPTY"
    assert be.assign_class(3, True, True) == "FOUND-BOTH"
    assert be.assign_class(3, False, True) == "FOUND-INDEPENDENT-ONLY"
    assert be.assign_class(3, True, False) == "FOUND-DB-ONLY"
    assert be.assign_class(3, False, False) == "ABSENT-BOTH"


# --- per-PDF health ---------------------------------------------------------------
def test_pdf_health_and_manifest_merge(tmp_path: Path) -> None:
    db = tmp_path / "corpus.db"
    fake_db(db)
    health = be.merge_manifest_health(be.pdf_health(db), fake_manifest())
    by = {h["rel_key"]: h for h in health}
    assert by["healthy.pdf"]["n_chunks"] == 1
    assert by["healthy.pdf"]["chars"] == 35
    assert by["healthy.pdf"]["pages_with_chunk"] == 1
    assert by["healthy.pdf"]["pages_with_no_chunk"] == [2]
    assert by["healthy.pdf"]["flag"] == "thin"  # 35 chars over 2 pages < 200/page
    assert by["slides.pdf"]["n_chunks"] == 0
    assert by["slides.pdf"]["flag"] == "zero-chunks"
    assert by["slides.pdf"]["avg_chars_per_page"] == 0.0
    assert by["thin.pdf"]["flag"] == ""  # 500 chars on 1 page >= 200/page
    assert by["thin.pdf"]["pages_with_no_chunk"] == []
    assert [h["rel_key"] for h in health] == ["healthy.pdf", "slides.pdf", "thin.pdf"]


# --- table integrity ----------------------------------------------------------------
def test_chunk_locality_same_chunk_passes() -> None:
    chunks = [
        {"text": "Revenue 5,000\nCost of goods sold 3,000"},
        {"text": "Net income 2,000"},
    ]
    out = be.chunk_locality(chunks, ["Revenue", "Cost of goods sold"])
    assert out["pass"] is True
    assert all(p["found"] and p["number_in_same_chunk"] for p in out["labels"])


def test_chunk_locality_label_split_from_its_value_fails() -> None:
    # the label sits in a chunk that carries no number: the table row was
    # split across a chunk boundary (a structure loss)
    chunks = [
        {"text": "Cost of goods sold"},
        {"text": "3,000\nNet income 2,000"},
    ]
    out = be.chunk_locality(chunks, ["Cost of goods sold"])
    assert out["pass"] is False
    lab = out["labels"][0]
    assert lab["found"] is True
    assert lab["number_in_same_chunk"] is False


def test_chunk_locality_label_missing_fails_and_no_labels_is_none() -> None:
    out = be.chunk_locality([{"text": "Revenue 5,000"}], ["EBITDA"])
    assert out["pass"] is False
    assert out["labels"][0]["found"] is False
    assert be.chunk_locality([{"text": "Revenue 5,000"}], [])["pass"] is None


def test_chunk_locality_label_in_numberless_and_numbered_chunks_passes() -> None:
    # a repeated label: a header row without numbers plus the real row
    chunks = [
        {"text": "Revenue"},
        {"text": "Revenue 5,000"},
    ]
    out = be.chunk_locality(chunks, ["Revenue"])
    assert out["pass"] is True
    assert out["labels"][0]["chunks"] == [0, 1]


def test_table_inputs_never_raises_on_a_dead_relay() -> None:
    class Dead:
        def post(self, payload: Any) -> dict[str, Any]:
            raise RuntimeError("relay down")

    computed, inputs, why, err = be.table_inputs(
        "What is the turnover ratio?", "3.46", Dead(), "model"
    )
    assert computed is False
    assert inputs == []
    assert "table-inputs call failed" in err


# --- the independent extractors on a generated PDF -----------------------------------
def test_independent_extractors_read_the_generated_pdf(tmp_path: Path) -> None:
    extractors = be.available_extractors()
    assert extractors, "the venv must have at least one of pypdf/pdfminer/pypdfium2"
    pdf = tmp_path / "one.pdf"
    make_one_page_pdf(pdf, "Total dividend 3,02,16,492 rupees FY 2021")
    for name, fn in extractors.items():
        text = fn(pdf, [1])
        found, _ = be.gold_in_text("30216492.00", "Float", text)
        assert found, f"{name} lost the number: {text[:120]!r}"
        found, _ = be.gold_in_text("FY2021", "Str", text)
        assert found, f"{name} lost the fiscal year: {text[:120]!r}"
    # a page beyond the document is skipped, not an error
    out = be.extract_pages(pdf, extractors, [1, 99])
    assert all(res["error"] is None for res in out.values())


def test_extract_pages_records_errors_without_raising(tmp_path: Path) -> None:
    pdf = tmp_path / "missing.pdf"

    def boom(path: Path, pages: list[int]) -> str:
        raise RuntimeError("corrupt page")

    out = be.extract_pages(pdf, {"boom": boom}, [1])
    assert out["boom"]["error"] == "corrupt page"
    assert out["boom"]["chars"] == 0


# --- the slide-deck verdict -------------------------------------------------------------
def test_deck_verdict_image_only_text_layer_and_error() -> None:
    empty = {
        "pypdf": {"error": None, "chars": 0, "first300": ""},
        "pdfminer": {"error": None, "chars": 3, "first300": "\f\f\f"},
    }
    assert be.deck_verdict(empty) == "image-only (no text layer)"
    with_text = {
        "pypdf": {"error": None, "chars": 500, "first300": "slide title ..."},
    }
    assert be.deck_verdict(with_text).startswith("has a text layer")
    only_error = {"pypdf": {"error": "boom", "chars": 0, "first300": ""}}
    assert be.deck_verdict(only_error) == "reader errored"
    assert be.deck_verdict({}) == "no extractor available"


def test_slide_deck_findings_on_the_fake_db(tmp_path: Path) -> None:
    # a zero-chunk doc whose (generated) PDF DOES carry text: the verdict is
    # 'has a text layer' - the ingest reader missed it
    db = tmp_path / "corpus.db"
    fake_db(db)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_one_page_pdf(corpus / "slides.pdf", "A slide title with enough text to count")
    decks = be.slide_deck_findings(db, corpus, be.available_extractors())
    assert [d["pdf"] for d in decks] == ["slides.pdf"]
    d = decks[0]
    assert d["pdf_local"] is True
    assert d["verdict"].startswith("has a text layer")
    assert all(res["chars"] >= be.DECK_MIN_CHARS for res in d["extractors"].values())
    # a zero-chunk doc with no PDF at all: recorded, not an error
    (corpus / "slides.pdf").unlink()
    decks = be.slide_deck_findings(db, corpus, be.available_extractors())
    assert decks[0]["pdf_local"] is False
    assert decks[0]["verdict"] == "no extractor available"
