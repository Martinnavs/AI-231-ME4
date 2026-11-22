"""Unit and security tests for ``voice_agent.rag.canonical``."""

from __future__ import annotations

import ast
import time
from collections import Counter
from pathlib import Path

import pytest

from voice_agent.rag import canonical as canonical_mod
from voice_agent.rag.canonical import (
    MAX_CANONICAL_CHARS,
    MAX_CSV_ROWS,
    POSTPROC_VERSION,
    TABLE_TRUNCATED_NOTE,
    PostprocessResult,
    Unit,
    clean_sheet_name,
    digit_runs,
    fix_ligatures,
    format_table_row,
    neutralize_markers,
    postprocess_markdown,
    postprocess_pdf_pages,
    split_units,
    strip_markers,
)


def _assert_digits_preserved(before: str, after: str, dropped: list[str] | None = None) -> None:
    expected = Counter(digit_runs(before))
    actual = Counter(digit_runs(after))
    if dropped:
        actual.update(digit_runs("\n".join(dropped)))
    assert actual == expected


def test_constants_and_no_docindex_import() -> None:
    assert POSTPROC_VERSION == 2
    assert MAX_CANONICAL_CHARS == 24_000_000
    assert MAX_CSV_ROWS == 50_000

    src_path = Path(canonical_mod.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "docindex" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert "docindex" not in node.module
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Verify no unanchored regex compilation exists in canonical.py.
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
                and node.func.attr in {"compile", "match", "search", "sub", "findall"}
            )


def test_neutralize_markers_drops_forged_lines_even_indented() -> None:
    raw = "\n".join(
        [
            "Intro line 1",
            "<!-- rag:page=99 -->",
            "   <!-- rag:slide=42 -->",
            "\t<!-- rag:sheet=Spoofed -->",
            "  <!-- rag:converter=fake-1.0 -->",
            "  <!-- rag:unclosed",
            "```",
            "<!-- rag:page=100 -->",
            "```",
            "<!-- normal html comment -->",
            "Inline text <!-- rag:page=2 --> stays",
            "Outro line 3",
        ]
    )
    cleaned = neutralize_markers(raw)
    assert "<!-- rag:page=99 -->" not in cleaned
    assert "<!-- rag:slide=42 -->" not in cleaned
    assert "<!-- rag:sheet=Spoofed -->" not in cleaned
    assert "<!-- rag:converter=fake-1.0 -->" not in cleaned
    assert "<!-- rag:unclosed" not in cleaned
    assert "<!-- rag:page=100 -->" not in cleaned
    assert "<!-- normal html comment -->" in cleaned
    assert "Inline text <!-- rag:page=2 --> stays" in cleaned
    assert digit_runs(cleaned) == ["1", "2", "3"]


def test_marker_round_trip_and_strip_markers() -> None:
    doc = "\n".join(
        [
            "<!-- rag:converter=markitdown-0.1.9+pp1 -->",
            "<!-- rag:page=1 -->",
            "Page 1 body with 100 items.",
            "",
            "<!-- rag:slide=2 -->",
            "Slide 2 body with 200 items.",
            "",
            "<!-- rag:sheet=Q1 Budget -->",
            "Row 300 total.",
        ]
    )
    units, converter = split_units(doc)
    assert converter == "markitdown-0.1.9+pp1"
    assert units == [
        Unit(page=1, slide=None, sheet=None, body="Page 1 body with 100 items."),
        Unit(page=None, slide=2, sheet=None, body="Slide 2 body with 200 items."),
        Unit(page=None, slide=None, sheet="Q1 Budget", body="Row 300 total."),
    ]
    for unit in units:
        assert "<!-- rag:" not in unit.body

    stripped = strip_markers(doc)
    assert "<!-- rag:" not in stripped
    assert stripped == (
        "Page 1 body with 100 items.\n\nSlide 2 body with 200 items.\n\nRow 300 total."
    )
    assert digit_runs(stripped) == ["1", "100", "2", "200", "300"]


def test_split_units_markerless_and_empty() -> None:
    assert split_units("") == ([], None)
    assert split_units("<!-- rag:converter=identity -->\n") == ([], "identity")
    units, converter = split_units("Plain text 42")
    assert converter is None
    assert units == [Unit(page=None, slide=None, sheet=None, body="Plain text 42")]
    assert digit_runs(units[0].body) == ["42"]


def test_sheet_name_cleaning() -> None:
    assert clean_sheet_name("  Revenue 2026  ") == "Revenue 2026"
    assert clean_sheet_name("A-->B") == "AB"
    assert clean_sheet_name("<Sheet>1<2>") == "Sheet12"
    assert clean_sheet_name("Line1\nLine2\r\nLine3") == "Line1Line2Line3"
    assert clean_sheet_name("  -->> <Budget\n2026>  ") == "Budget2026"
    long_name = "  " + ("S" * 150) + "  "
    cleaned_long = clean_sheet_name(long_name)
    assert len(cleaned_long) == 120
    assert cleaned_long == "S" * 120

    md = "##   Q1 --> <Summary>\n\n| Item | Cost |\n| --- | --- |\n| Pen | 5 |\n"
    res = postprocess_markdown(md, kind="xlsx")
    assert "<!-- rag:sheet=Q1  Summary -->" in res.text
    units, _ = split_units(res.text)
    assert len(units) == 1
    assert units[0].sheet == "Q1  Summary"
    assert digit_runs(units[0].body) == ["5"]


def test_ligatures_normalized_and_digits_preserved() -> None:
    ligatures = "ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ"
    expected = "fi fl ff ffi ffl st st"
    assert fix_ligatures(ligatures) == expected
    assert fix_ligatures("oﬃce ﬀ 123") == "office ff 123"
    assert digit_runs("oﬃce 123") == digit_runs(fix_ligatures("oﬃce 123"))


@pytest.mark.parametrize(
    ("case_name", "sample", "expected_text", "expected_digits"),
    [
        ("superscript_10_2", "10²", "10²", ["10"]),
        ("superscript_m_3", "m³", "m³", []),
        ("fraction_half", "½", "½", []),
        ("fraction_quarter", "¼", "¼", []),
        ("ligatures_all", "ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ", "fi fl ff ffi ffl st st", []),
        ("soft_hyphen_removed", "co\u00adop 2026", "coop 2026", ["2026"]),
        ("nbsp_preserved", "100\u00a0kg", "100\u00a0kg", ["100"]),
        (
            "en_and_em_dash",
            "2020–2026 and alpha—beta",
            "2020–2026 and alpha—beta",
            ["2020", "2026"],
        ),
        ("minus_sign", "−42", "−42", ["42"]),
        ("fullwidth_digits", "１２３", "１２３", ["123"]),
        ("currency_symbols", "$ € £ ¥ 99", "$ € £ ¥ 99", ["99"]),
        ("footnote_superscript", "word¹", "word¹", []),
        ("footnote_bracket", "[1]", "[1]", ["1"]),
        ("footnote_asterisk", "*", "*", []),
    ],
)
def test_edge_case_list(
    case_name: str,
    sample: str,
    expected_text: str,
    expected_digits: list[str],
) -> None:
    assert digit_runs(sample) == expected_digits

    # fix_ligatures must leave non-FB00..FB06 inputs byte-unchanged (incl. U+00AD).
    lig_out = fix_ligatures(sample)
    if case_name == "ligatures_all":
        assert lig_out == expected_text
    else:
        assert lig_out == sample
    assert digit_runs(sample) == digit_runs(lig_out)

    # Full postprocessors preserve non-ligature/non-soft-hyphen edge cases unchanged.
    pdf_res = postprocess_pdf_pages([sample])
    assert isinstance(pdf_res, PostprocessResult)
    pdf_body = strip_markers(pdf_res.text)
    assert pdf_body == expected_text
    assert digit_runs(sample) == digit_runs(pdf_body)

    md_res = postprocess_markdown(sample, kind="md")
    assert md_res.text == expected_text
    assert digit_runs(sample) == digit_runs(md_res.text)


def test_fix_ligatures_bytes_unchanged_for_superscript_and_fraction() -> None:
    assert fix_ligatures("10²").encode("utf-8") == b"10\xc2\xb2"
    assert fix_ligatures("½").encode("utf-8") == b"\xc2\xbd"


def test_postprocess_pdf_pages_4page_header_footer_drop_and_digit_accounting() -> None:
    pages = [
        "Acme Corp Confidential\n\nFirst page body 100.\nExtra line.\nPage 1 of 4",
        (
            "Acme Corp Confidential\n\nSecond page body 200.\n"
            "Middle Acme Corp Confidential\nPage 2 of 4"
        ),
        "Acme Corp Confidential\n\nThird page body 300.\nMore text.\nPage 3 of 4",
        "Acme Corp Confidential\n\nFourth page body 400.\nClosing.\nPage 4 of 4",
    ]
    res = postprocess_pdf_pages(pages)
    assert "<!-- rag:page=1 -->" in res.text
    assert "<!-- rag:page=4 -->" in res.text
    assert "Page 1 of 4" not in res.text
    assert "Page 4 of 4" not in res.text
    # Middle line matching header text is NOT in first/last 2 non-empty lines -> preserved.
    assert "Middle Acme Corp Confidential" in res.text
    assert len(res.dropped_lines) == 8
    assert "Acme Corp Confidential" in res.dropped_lines
    assert "Page 1 of 4" in res.dropped_lines
    assert "Page 4 of 4" in res.dropped_lines

    before_all = "\n".join(pages)
    _assert_digits_preserved(before_all, strip_markers(res.text), res.dropped_lines)


def test_postprocess_pdf_pages_2page_header_footer_is_noop() -> None:
    pages = [
        "Employee Handbook\nWelcome to page 1.\nPage 1",
        "Employee Handbook\nWelcome to page 2.\nPage 2",
    ]
    res = postprocess_pdf_pages(pages)
    assert res.dropped_lines == []
    assert "Employee Handbook" in res.text
    assert "Page 1" in res.text
    assert "Page 2" in res.text
    _assert_digits_preserved("\n".join(pages), strip_markers(res.text))


def test_postprocess_pdf_pages_dehyphenation_whitespace_and_blank_pages() -> None:
    pages = [
        "",
        "hy-\nphen\u00adated   text\t\there  \n\n\n\nWrapped\nline stays 10.\nend-\n",
        "next 20.",
    ]
    res = postprocess_pdf_pages(pages)
    assert "<!-- rag:page=1 -->" not in res.text
    assert "<!-- rag:page=2 -->" in res.text
    assert "<!-- rag:page=3 -->" in res.text
    assert "hyphenated text here" in res.text
    # Wrapped lines are NOT joined (non-goal).
    assert "Wrapped\nline stays 10." in res.text
    # Dehyphenation across pages is NOT performed (non-goal).
    assert "end-" in res.text
    assert "\n\n\n" not in res.text
    _assert_digits_preserved("\n".join(pages), strip_markers(res.text), res.dropped_lines)


def test_postprocess_markdown_pptx_and_xlsx_markers() -> None:
    pptx_md = (
        "\n\n<!-- Slide number: 1 -->\n# Title 10\n\nBullet   one\n\n"
        "<!-- Slide number: 2 -->\n# Next 20\n"
    )
    res = postprocess_markdown(pptx_md, kind="pptx")
    units, conv = split_units(res.text)
    assert conv is None
    assert len(units) == 2
    assert units[0] == Unit(page=None, slide=1, sheet=None, body="# Title 10\n\nBullet one")
    assert units[1] == Unit(page=None, slide=2, sheet=None, body="# Next 20")
    assert digit_runs(strip_markers(res.text)) == ["10", "20"]


def test_postprocess_markdown_tables_and_fences() -> None:
    md = "\n".join(
        [
            "| | Item | Note | Price |",
            "| :--- | :---: | ---: | --- |",
            r"| A1 | Widget \| Pro | NaN | 19.99 |",
            "| B2 |  | nan | 25 |",
            "| | | NaN | |",
            r"| C3 | Path\\|Val | 30 |",
            "",
            "```markdown",
            "| Keep | Raw |",
            "| --- | --- |",
            "| 1   2 |   NaN |",
            "```",
            "",
            "~~~",
            "## Keep heading   raw",
            "~~~",
            "",
            "Not a | table without separator",
        ]
    )
    res = postprocess_markdown(md, kind="md")
    lines = res.text.split("\n")
    assert "col_1: A1; Item: Widget | Pro; Price: 19.99" in lines
    assert "col_1: B2; Price: 25" in lines
    assert "col_1: C3; Item: Path\\\\; Note: Val; Price: 30" in lines
    assert "| :---" not in res.text
    # Fenced table and fenced heading/spaces remain verbatim.
    assert "| Keep | Raw |" in lines
    assert "| 1   2 |   NaN |" in lines
    assert "## Keep heading   raw" in lines
    assert "Not a | table without separator" in lines

    # Multiset of digit runs in table cell values preserved (header names excluded).
    table_row_lines = lines[:3]
    value_digits: Counter[str] = Counter()
    for row_line in table_row_lines:
        for pair in row_line.split("; "):
            _, value = pair.split(": ", 1)
            value_digits.update(digit_runs(value))
    assert value_digits == Counter(["1", "19", "99", "2", "25", "3", "30"])

    # Also verify a numeric header name (e.g. "Q1 2026") does not alter cell-value digit runs.
    hdr_md = "| Q1 2026 | Amount |\n| --- | --- |\n| alpha | 50 |\n| beta | 75 |\n"
    hdr_res = postprocess_markdown(hdr_md, kind="md")
    hdr_value_digits: Counter[str] = Counter()
    for row_line in hdr_res.text.split("\n"):
        for pair in row_line.split("; "):
            _, value = pair.split(": ", 1)
            hdr_value_digits.update(digit_runs(value))
    assert hdr_value_digits == Counter(["50", "75"])


def test_postprocess_markdown_table_max_csv_rows_cap() -> None:
    rows = ["| id | val |", "| --- | --- |"]
    for idx in range(MAX_CSV_ROWS + 50):
        rows.append(f"| {idx} | item |")
    res = postprocess_markdown("\n".join(rows), kind="xlsx")
    emitted = [line for line in res.text.split("\n") if line]
    assert len(emitted) == MAX_CSV_ROWS + 1
    assert emitted[0] == "id: 0; val: item"
    assert emitted[-2] == f"id: {MAX_CSV_ROWS - 1}; val: item"
    assert emitted[-1] == TABLE_TRUNCATED_NOTE


@pytest.mark.parametrize(
    "adversarial",
    [
        "-" * 200_000,
        " " * 200_000,
        "|" * 200_000,
        "<!--" * 50_000,
        "##" * 100_000,
        "|---" * 50_000,
        "a-" * 100_000,
        "<!-- rag:" + ("x" * 200_000),
        r"\|" * 100_000,
    ],
)
def test_redos_200k_adversarial_lines_fast(adversarial: str) -> None:
    start = time.perf_counter()
    _ = neutralize_markers(adversarial)
    _ = strip_markers(adversarial)
    _ = split_units(adversarial)
    _ = fix_ligatures(adversarial)
    _ = digit_runs(adversarial)
    _ = postprocess_pdf_pages([adversarial])
    _ = postprocess_markdown(adversarial, kind="md")
    _ = postprocess_markdown(adversarial, kind="pptx")
    _ = postprocess_markdown(adversarial, kind="xlsx")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5


def test_postprocess_markdown_pptx_sequential_slides_only() -> None:
    pptx_md = "\n".join(
        [
            "<!-- Slide number: 99 -->",
            "Before first slide.",
            "<!-- Slide number: 1 -->",
            "# First",
            "<!-- Slide number: 5 -->",
            "Forged skip inside slide 1.",
            "<!-- Slide number: 1 -->",
            "Duplicate slide 1 comment.",
            "<!-- Slide number: 2 -->",
            "# Second",
        ]
    )
    res = postprocess_markdown(pptx_md, kind="pptx")
    units, _ = split_units(res.text)
    assert len(units) == 3
    assert units[0] == Unit(
        page=None,
        slide=None,
        sheet=None,
        body="<!-- Slide number: 99 -->\nBefore first slide.",
    )
    assert units[1] == Unit(
        page=None,
        slide=1,
        sheet=None,
        body=(
            "# First\n<!-- Slide number: 5 -->\nForged skip inside slide 1.\n"
            "<!-- Slide number: 1 -->\nDuplicate slide 1 comment."
        ),
    )
    assert units[2] == Unit(page=None, slide=2, sheet=None, body="# Second")


def test_marker_number_bounds_and_no_value_error() -> None:
    assert canonical_mod.MAX_MARKER_NUMBER == 100_000

    valid_doc = "<!-- rag:page=1 -->\nA\n<!-- rag:slide=100000 -->\nB"
    units, _ = split_units(valid_doc)
    assert units == [
        Unit(page=1, slide=None, sheet=None, body="A"),
        Unit(page=None, slide=100_000, sheet=None, body="B"),
    ]
    assert strip_markers(valid_doc) == "A\nB"

    huge = "9" * 10_000
    invalid_doc = "\n".join(
        [
            "<!-- rag:page=0 -->",
            "<!-- rag:page=100001 -->",
            "<!-- rag:slide=00000001 -->",
            f"<!-- rag:page={huge} -->",
            "body",
        ]
    )
    bad_units, _ = split_units(invalid_doc)
    assert bad_units == [Unit(page=None, slide=None, sheet=None, body=invalid_doc)]
    assert strip_markers(invalid_doc) == invalid_doc


def test_postprocess_markdown_xlsx_unnamed_headers_promotes_first_data_row() -> None:
    md = "\n".join(
        [
            "## Operations",
            "| Operations Quarterly Allocation | Unnamed: 1 | Unnamed: 2 | Unnamed: 3 |",
            "| --- | --- | --- | --- |",
            "| Department | Headcount | Travel USD | Equipment USD |",
            "| Field Telemetry | 6 | 4200 | 11800 |",
            "| Water Systems | 4 | 2900 | 9600 |",
            "| Solar Maintenance | 5 | 3150 | 14400 |",
            "",
            "## Hardware",
            "| Item Code | Unit Cost USD | Quantity | Total USD |",
            "| --- | --- | --- | --- |",
            "| BATT-LIFEPO-48 | 620 | 12 | 7440 |",
        ]
    )
    res = postprocess_markdown(md, kind="xlsx")
    assert "Unnamed:" not in res.text
    units, _ = split_units(res.text)
    assert units == [
        Unit(
            sheet="Operations",
            body="\n".join(
                [
                    "Operations Quarterly Allocation | Department: Field Telemetry; "
                    "Headcount: 6; Travel USD: 4200; Equipment USD: 11800",
                    "Department: Water Systems; Headcount: 4; Travel USD: 2900; "
                    "Equipment USD: 9600",
                    "Department: Solar Maintenance; Headcount: 5; Travel USD: 3150; "
                    "Equipment USD: 14400",
                ]
            ),
        ),
        Unit(
            sheet="Hardware",
            body="Item Code: BATT-LIFEPO-48; Unit Cost USD: 620; Quantity: 12; Total USD: 7440",
        ),
    ]


def test_postprocess_markdown_xlsx_all_blank_headers_promotes_without_title() -> None:
    md = "\n".join(
        [
            "## Roster",
            "| Unnamed: 0 | Unnamed: 1 | Unnamed: 2 |",
            "| --- | --- | --- |",
            "| Name | Role | Shift |",
            "| Ana | Lead | Day |",
        ]
    )
    res = postprocess_markdown(md, kind="xlsx")
    assert "Unnamed:" not in res.text
    units, _ = split_units(res.text)
    assert units == [Unit(sheet="Roster", body="Name: Ana; Role: Lead; Shift: Day")]


def test_postprocess_markdown_xlsx_no_promotion_partial_or_blank_row1() -> None:
    less_than_half = "\n".join(
        [
            "## Partial",
            "| Item | Unnamed: 1 | Qty | Cost |",
            "| --- | --- | --- | --- |",
            "| Bolt | M6 | 10 | 5 |",
        ]
    )
    res_lt = postprocess_markdown(less_than_half, kind="xlsx")
    assert "Unnamed:" not in res_lt.text
    units_lt, _ = split_units(res_lt.text)
    assert units_lt == [
        Unit(sheet="Partial", body="Item: Bolt; col_2: M6; Qty: 10; Cost: 5"),
    ]

    row1_has_blank = "\n".join(
        [
            "## Sparse",
            "| Title | Unnamed: 1 | Unnamed: 2 |",
            "| --- | --- | --- |",
            "| Group | NaN | 100 |",
            "| Alpha | Beta | 200 |",
        ]
    )
    res_sparse = postprocess_markdown(row1_has_blank, kind="xlsx")
    assert "Unnamed:" not in res_sparse.text
    units_sparse, _ = split_units(res_sparse.text)
    assert units_sparse == [
        Unit(
            sheet="Sparse",
            body="Title: Group; col_3: 100\nTitle: Alpha; col_2: Beta; col_3: 200",
        ),
    ]


def test_postprocess_markdown_non_xlsx_tables_unchanged_with_unnamed_or_blank_headers() -> None:
    table_md = "\n".join(
        [
            "| Title | Unnamed: 1 | Unnamed: 2 |",
            "| --- | --- | --- |",
            "| ColA | ColB | ColC |",
            "| 1 | 2 | 3 |",
        ]
    )
    for kind in ("md", "pptx", "html", "docx"):
        res = postprocess_markdown(table_md, kind=kind)
        assert res.text == "\n".join(
            [
                "Title: ColA; Unnamed: 1: ColB; Unnamed: 2: ColC",
                "Title: 1; Unnamed: 1: 2; Unnamed: 2: 3",
            ]
        )


def test_postprocess_markdown_xlsx_unnamed_header_redos_linear_time() -> None:
    wide_unnamed = "Unnamed: " + ("9" * 200_000)
    md = f"## Wide\n| {wide_unnamed} | Unnamed: 1 |\n| --- | --- |\n| H1 | H2 |\n| A | 10 |"
    start = time.perf_counter()
    res = postprocess_markdown(md, kind="xlsx")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5
    assert "Unnamed:" not in res.text
    assert "H1: A; H2: 10" in res.text


@pytest.mark.parametrize(
    ("header", "expected_prefix"),
    [
        ("# Qty", "\\# Qty: 3"),
        ("```", "\\```: 3"),
        ("~~~", "\\~~~: 3"),
        ("<!-- rag:x", "\\<!-- rag:x: 3"),
        ("<!-- Slide number: 2", "\\<!-- Slide number: 2: 3"),
    ],
)
def test_format_table_row_escapes_line_start(header: str, expected_prefix: str) -> None:
    line = format_table_row([header, "Status"], ["3", "#N/A"])
    assert line.startswith(expected_prefix)
    assert line.endswith("; Status: #N/A")


def test_format_table_row_plain_rows_unchanged_and_not_double_escaped() -> None:
    assert format_table_row(["Date", "A-"], ["1", "x"]) == "Date: 1; A-: x"
    assert format_table_row(["Qty"], ["#N/A"]) == "Qty: #N/A"
    assert format_table_row(["\\# Qty"], ["3"]) == "\\# Qty: 3"
