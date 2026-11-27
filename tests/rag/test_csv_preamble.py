"""A title or preamble line above a csv header is not the header (ticket 254)."""

from __future__ import annotations

from voice_agent.rag.docindex import parse_csv_text


def test_a_one_cell_title_row_is_kept_and_the_next_row_is_the_header() -> None:
    text = "Quarterly output\nName,2020,2021\nalpha,5,6\nbeta,7,8\n"
    out = parse_csv_text(text).split("\n")
    assert out[0] == "Quarterly output"
    assert out[1] == "Name: alpha; 2020: 5; 2021: 6"
    assert out[2] == "Name: beta; 2020: 7; 2021: 8"
    assert "col_2" not in "\n".join(out)


def test_several_title_lines_and_blank_lines_above_the_header() -> None:
    text = "Report A\n\nPrepared for x\nid,qty\n1,9\n"
    out = parse_csv_text(text).split("\n")
    assert out[:2] == ["Report A", "Prepared for x"]
    assert out[2] == "id: 1; qty: 9"


def test_a_normal_csv_is_unchanged() -> None:
    text = "a,b,c\n1,2,3\n4,,6\n"
    assert parse_csv_text(text) == "a: 1; b: 2; c: 3\na: 4; c: 6"


def test_a_header_with_blank_cells_but_several_filled_is_still_the_header() -> None:
    text = "name,,qty\nx,y,3\n"
    assert parse_csv_text(text) == "name: x; col_2: y; qty: 3"


def test_a_one_column_file_is_unchanged() -> None:
    text = "items\napple\npear\n"
    assert parse_csv_text(text) == "items: apple\nitems: pear"


def test_no_table_within_the_look_ahead_leaves_the_file_as_it_was() -> None:
    text = "t\n" + "x\n" * 10 + "a,b\n1,2\n"
    out = parse_csv_text(text)
    assert out.splitlines()[0] == "t: x"
