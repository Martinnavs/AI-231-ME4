"""Tests for zip/XML preflight guards in ``voice_agent.rag.zipguard`` and ``load_docx``."""

from __future__ import annotations

import copy
import io
import struct
import time
import warnings
import zipfile

import pytest

from voice_agent.rag import docindex, zipguard


def _make_zip(
    members: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    allow_duplicates: bool = False,
) -> bytes:
    buf = io.BytesIO()
    cm = warnings.catch_warnings()
    with cm:
        if allow_duplicates:
            warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buf, "w", compression=compression) as zf:
            for name, data in members:
                zf.writestr(name, data)
    return buf.getvalue()


def _nested_xml(depth: int, root_tag: str = "root", child_tag: str = "n") -> bytes:
    inner = "<leaf>ok</leaf>"
    # root(1) + (depth - 2) * child + leaf(depth) == depth
    nesting = depth - 2
    assert nesting >= 0
    open_tags = f"<{root_tag}>" + "".join(f'<{child_tag} id="{i}">' for i in range(nesting))
    close_tags = f"</{child_tag}>" * nesting + f"</{root_tag}>"
    return f'<?xml version="1.0" encoding="UTF-8"?>{open_tags}{inner}{close_tags}'.encode()


def test_clean_multi_member_zip_passes() -> None:
    raw = _make_zip(
        [
            ("word/", b""),
            ("_rels/.rels", b'<?xml version="1.0"?><Relationships/>'),
            ("word/document.xml", b'<?xml version="1.0"?><doc><p>Hello</p></doc>'),
            ("word/styles.xml", _nested_xml(25, root_tag="styles")),
            ("OEBPS/content.opf", b'<?xml version="1.0"?><package/>'),
            ("OEBPS/toc.ncx", b'<?xml version="1.0"?><ncx/>'),
            ("OEBPS/ch1.xhtml", b'<?xml version="1.0"?><html/>'),
            ("OEBPS/ch2.html", b"<html><body><p>ok</p></body></html>"),
            ("OEBPS/ch3.htm", b"<html><body><p>ok</p></body></html>"),
            ("word/media/image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
        ]
    )
    assert zipguard.preflight_zip(raw, label="epub", scan_xml_members=True) is None
    assert zipguard.preflight_zip(raw, label="docx", scan_xml_members=False) is None


@pytest.mark.parametrize(
    "member_name",
    [
        "word/styles.xml",
        "_rels/.rels",
        "OEBPS/content.opf",
        "OEBPS/toc.ncx",
        "OEBPS/chapter1.xhtml",
        "OEBPS/chapter1.html",
        "OEBPS/chapter1.htm",
        "WORD/UPPER.XML",
    ],
)
def test_doctype_in_non_document_member_rejected_when_scanning(member_name: str) -> None:
    raw = _make_zip(
        [
            ("word/document.xml", b'<?xml version="1.0"?><doc><p>Clean</p></doc>'),
            (member_name, b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY x "1">]><root/>'),
        ]
    )
    err = zipguard.preflight_zip(raw, label="pptx", scan_xml_members=True)
    assert err is not None
    assert "pptx" in err
    assert "DOCTYPE" in err

    # With scan_xml_members=False, non-document XML members are not inflated/scanned.
    assert zipguard.preflight_zip(raw, label="pptx", scan_xml_members=False) is None


def test_entity_and_utf16_decl_in_non_document_member_rejected() -> None:
    entity_zip = _make_zip(
        [
            ("word/document.xml", b'<?xml version="1.0"?><doc/>'),
            ("ppt/slides/slide1.xml", b'<?xml version="1.0"?><!ENTITY xxe "boom"><sld/>'),
        ]
    )
    err = zipguard.preflight_zip(entity_zip, label="pptx", scan_xml_members=True)
    assert err == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"

    utf16_payload = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x><sld/>'.encode("utf-16")
    utf16_zip = _make_zip(
        [
            ("word/document.xml", b'<?xml version="1.0"?><doc/>'),
            ("ppt/slides/slide2.xml", utf16_payload),
        ]
    )
    err_u16 = zipguard.preflight_zip(utf16_zip, label="pptx", scan_xml_members=True)
    assert err_u16 == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"


def test_150_deep_non_document_member_rejected() -> None:
    raw = _make_zip(
        [
            ("word/document.xml", b'<?xml version="1.0"?><doc><p>Clean</p></doc>'),
            ("xl/worksheets/sheet1.xml", _nested_xml(150, root_tag="worksheet")),
        ]
    )
    err = zipguard.preflight_zip(raw, label="xlsx", scan_xml_members=True)
    assert err == f"xlsx XML is nested deeper than {zipguard.MAX_DOCX_XML_DEPTH} levels"
    assert zipguard.preflight_zip(raw, label="xlsx", scan_xml_members=False) is None


def test_label_appears_in_all_zipguard_error_messages() -> None:
    # 1. Too many members.
    many = _make_zip(
        [(f"m_{i}.txt", b"a") for i in range(6)],
        compression=zipfile.ZIP_STORED,
    )
    err = zipguard.preflight_zip(many, label="pptx", max_members=5, scan_xml_members=False)
    assert err == "pptx has too many zip members (6 > 5)"

    # 2. Declared member bytes exceeded.
    big_member = _make_zip([("a.xml", b"<a/>" * 50)], compression=zipfile.ZIP_STORED)
    err = zipguard.preflight_zip(
        big_member, label="xlsx", max_member_bytes=50, scan_xml_members=False
    )
    assert err == "xlsx member 'a.xml' exceeds 50 uncompressed bytes"

    # 3. Declared total bytes exceeded.
    big_total = _make_zip(
        [("a.bin", b"A" * 60), ("b.bin", b"B" * 60)],
        compression=zipfile.ZIP_STORED,
    )
    err = zipguard.preflight_zip(
        big_total, label="epub", max_total_bytes=100, scan_xml_members=False
    )
    assert err == "epub total uncompressed size exceeds 100 bytes"

    # 4. Overlapping zip entries.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.xml", b"<root>content</root>")
        orig = zf.getinfo("a.xml")
        dup = copy.copy(orig)
        dup.filename = "b.xml"
        zf.filelist.append(dup)
    err = zipguard.preflight_zip(buf.getvalue(), label="pptx", scan_xml_members=False)
    assert err == "pptx has overlapping zip entries"

    # 5. Invalid/corrupt zip.
    err = zipguard.preflight_zip(b"not a zip archive", label="epub", scan_xml_members=False)
    assert err is not None
    assert err.startswith("invalid epub archive:")

    # 6. Malformed XML in scanned member.
    bad_xml = _make_zip([("OEBPS/bad.xhtml", b"<root><unclosed></root>")])
    err = zipguard.preflight_zip(bad_xml, label="epub", scan_xml_members=True)
    assert err is not None
    assert err.startswith("epub XML parse error:")

    # 7. Empty XML in scanned member.
    empty_xml = _make_zip([("OEBPS/empty.xml", b"")])
    err = zipguard.preflight_zip(empty_xml, label="epub", scan_xml_members=True)
    assert err is not None
    assert err.startswith("epub XML parse error:")


def test_scan_xml_members_enforces_total_cap_and_rejects_forged_size() -> None:
    xml_a = b"<root>" + b"<x>0123456789</x>" * 30 + b"</root>"
    xml_b = b"<root>" + b"<y>0123456789</y>" * 30 + b"</root>"
    raw = _make_zip([("a.xml", xml_a), ("b.xml", xml_b)], compression=zipfile.ZIP_DEFLATED)
    err = zipguard.preflight_zip(
        raw,
        label="pptx",
        scan_xml_members=True,
        max_member_bytes=600,
        max_total_bytes=700,
    )
    assert err == "pptx total uncompressed size exceeds 700 bytes"

    # Forging uncompressed size down in headers is caught during scan_xml_members=True.
    deflated = bytearray(raw)
    with zipfile.ZipFile(io.BytesIO(deflated)) as zf:
        for info in zf.infolist():
            deflated[info.header_offset + 22 : info.header_offset + 26] = struct.pack("<I", 10)
    pos = 0
    while True:
        cd_idx = deflated.find(b"PK\x01\x02", pos)
        if cd_idx == -1:
            break
        deflated[cd_idx + 24 : cd_idx + 28] = struct.pack("<I", 10)
        pos = cd_idx + 4

    forged_err = zipguard.preflight_zip(
        bytes(deflated),
        label="pptx",
        scan_xml_members=True,
        max_member_bytes=600,
        max_total_bytes=700,
    )
    assert forged_err is not None
    assert "pptx" in forged_err


def test_direct_zipguard_helpers() -> None:
    raw = _make_zip([("a.xml", b"<root>hello</root>")], compression=zipfile.ZIP_STORED)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        info = zf.getinfo("a.xml")
        assert zipguard.read_member_bounded(zf, info, 100, keep=True, label="pptx") == (
            b"<root>hello</root>"
        )
        assert zipguard.read_member_bounded(zf, info, 100, keep=False, label="pptx") == b""
        with pytest.raises(ValueError, match="pptx member 'a.xml' exceeds 5 uncompressed bytes"):
            zipguard.read_member_bounded(zf, info, 5, keep=True, label="pptx")

    assert zipguard.has_forbidden_xml_decl(b"<!DOCTYPE x><r/>") is True
    assert zipguard.has_forbidden_xml_decl(b"<!entity x 'y'><r/>") is True
    assert zipguard.has_forbidden_xml_decl(b"<r>ok</r>") is False

    root, depth_err = zipguard.parse_xml_bounded(_nested_xml(5), 10, label="xlsx")
    assert root is not None
    assert depth_err is None

    root_deep, depth_err_deep = zipguard.parse_xml_bounded(_nested_xml(12), 10, label="xlsx")
    assert root_deep is None
    assert depth_err_deep == "xlsx XML is nested deeper than 10 levels"


def test_load_docx_error_strings_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_ok = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>"
    )

    # Too many members.
    monkeypatch.setattr(docindex, "MAX_DOCX_MEMBERS", 2)
    raw_many = _make_zip(
        [("word/document.xml", doc_ok), ("a.txt", b"1"), ("b.txt", b"2")],
        compression=zipfile.ZIP_STORED,
    )
    assert docindex.load_docx(raw_many) == (
        "",
        "unreadable",
        "docx has too many zip members (3 > 2)",
    )
    monkeypatch.undo()

    # Duplicate member name.
    raw_dup = _make_zip(
        [("word/document.xml", doc_ok), ("word/document.xml", doc_ok)],
        allow_duplicates=True,
    )
    assert docindex.load_docx(raw_dup) == (
        "",
        "unreadable",
        "duplicate zip member name: 'word/document.xml'",
    )

    # Compression ratio.
    raw_ratio = _make_zip(
        [("word/document.xml", doc_ok), ("bomb.txt", b"A" * 50_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert docindex.load_docx(raw_ratio) == (
        "",
        "unreadable",
        "suspicious compression ratio in 'bomb.txt'",
    )

    # Member size limit.
    monkeypatch.setattr(docindex, "MAX_DOCX_MEMBER_BYTES", 50)
    raw_member = _make_zip([("word/document.xml", doc_ok)], compression=zipfile.ZIP_STORED)
    assert docindex.load_docx(raw_member) == (
        "",
        "unreadable",
        "docx member 'word/document.xml' exceeds 50 uncompressed bytes",
    )
    monkeypatch.undo()

    # Total size limit.
    monkeypatch.setattr(docindex, "MAX_DOCX_TOTAL_BYTES", 100)
    raw_total = _make_zip(
        [("word/document.xml", doc_ok), ("extra.bin", b"X" * 100)],
        compression=zipfile.ZIP_STORED,
    )
    assert docindex.load_docx(raw_total) == (
        "",
        "unreadable",
        "docx total uncompressed size exceeds 100 bytes",
    )
    monkeypatch.undo()

    # Missing word/document.xml.
    raw_missing = _make_zip([("other.xml", b"<root/>")])
    assert docindex.load_docx(raw_missing) == ("", "unreadable", "missing word/document.xml")

    # Overlap.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", doc_ok)
        dup = copy.copy(zf.getinfo("word/document.xml"))
        dup.filename = "word/overlap.xml"
        zf.filelist.append(dup)
    assert docindex.load_docx(buf.getvalue()) == (
        "",
        "unreadable",
        "docx has overlapping zip entries",
    )

    # Invalid zip.
    status_tuple = docindex.load_docx(b"not a zip")
    assert status_tuple[0] == ""
    assert status_tuple[1] == "unreadable"
    assert status_tuple[2] is not None
    assert status_tuple[2].startswith("invalid docx archive: ")

    # Forbidden DOCTYPE / ENTITY.
    raw_doctype = _make_zip(
        [("word/document.xml", b'<?xml version="1.0"?><!DOCTYPE foo><w:document/>')]
    )
    assert docindex.load_docx(raw_doctype) == (
        "",
        "unreadable",
        "docx XML contains forbidden DOCTYPE or ENTITY declaration",
    )

    # Depth exceeded.
    raw_deep = _make_zip([("word/document.xml", _nested_xml(105))])
    assert docindex.load_docx(raw_deep) == (
        "",
        "unreadable",
        "docx XML is nested deeper than 100 levels",
    )

    # Malformed XML.
    raw_bad_xml = _make_zip([("word/document.xml", b"<w:document><broken>")])
    bad_res = docindex.load_docx(raw_bad_xml)
    assert bad_res[0] == ""
    assert bad_res[1] == "unreadable"
    assert bad_res[2] is not None
    assert bad_res[2].startswith("docx XML parse error: ")

    # Empty docx text.
    raw_empty_doc = _make_zip(
        [
            (
                "word/document.xml",
                b'<?xml version="1.0"?>'
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
            )
        ]
    )
    assert docindex.load_docx(raw_empty_doc) == ("", "empty", "docx contains no text")


def _make_minimal_epub_bytes(chapter_xhtml: bytes, *, ncx_bytes: bytes | None = None) -> bytes:
    container_xml = (
        b'<?xml version="1.0"?>'
        b'<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        b"<rootfiles>"
        b'<rootfile full-path="OEBPS/content.opf" '
        b'media-type="application/oebps-package+xml"/>'
        b"</rootfiles>"
        b"</container>"
    )
    content_opf = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        b'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        b"<dc:title>Guide</dc:title><dc:language>en</dc:language>"
        b"</metadata>"
        b"<manifest>"
        b'<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        b"</manifest>"
        b'<spine><itemref idref="ch1"/></spine>'
        b"</package>"
    )
    members: list[tuple[str, bytes]] = [
        ("mimetype", b"application/epub+zip"),
        ("META-INF/container.xml", container_xml),
        ("OEBPS/content.opf", content_opf),
        ("OEBPS/chapter1.xhtml", chapter_xhtml),
    ]
    if ncx_bytes is not None:
        members.append(("OEBPS/toc.ncx", ncx_bytes))
    return _make_zip(members)


@pytest.mark.parametrize(
    "payload",
    [
        b'<?xml version="1.0"?><!DOCTYPE worksheet><worksheet/>',
        b'<?xml version="1.0"?><!DOCTYPE html><html/>',
        b'\xef\xbb\xbf  \n<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "1">]><r/>',
        '<?xml version="1.0"?><!ENTITY xxe "boom"><r/>'.encode("utf-16"),
    ],
)
def test_renamed_member_starting_with_lt_and_doctype_or_entity_rejected(payload: bytes) -> None:
    raw = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("xl/worksheets/sheet1.dat", payload),
        ]
    )
    err = zipguard.preflight_zip(raw, label="xlsx", scan_xml_members=True)
    assert err == "xlsx XML contains forbidden DOCTYPE or ENTITY declaration"


def test_epub_plain_doctype_and_nbsp_pass_guard_while_entity_and_subset_rejected() -> None:
    xhtml_ok = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b"<!DOCTYPE html>\n"
        b'<html xmlns="http://www.w3.org/1999/xhtml">'
        b"<head><title>Ch 1</title></head>"
        b"<body><h1>Heading</h1><p>Sensor&nbsp;reading&mdash;ok.</p></body>"
        b"</html>"
    )
    ncx_ok = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" '
        b'"http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">\n'
        b'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap/></ncx>'
    )
    epub_ok = _make_minimal_epub_bytes(xhtml_ok, ncx_bytes=ncx_ok)
    assert zipguard.preflight_zip(epub_ok, label="epub", scan_xml_members=True) is None

    # ENTITY declaration in XHTML must still be rejected.
    xhtml_entity = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<!ENTITY xxe "boom">\n'
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Bad</p></body></html>'
    )
    epub_entity = _make_minimal_epub_bytes(xhtml_entity)
    assert (
        zipguard.preflight_zip(epub_entity, label="epub", scan_xml_members=True)
        == "epub XML contains forbidden DOCTYPE or ENTITY declaration"
    )

    # DOCTYPE with an internal subset ([) must still be rejected even without ENTITY.
    xhtml_subset = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b"<!DOCTYPE html [<!ELEMENT html ANY>]>\n"
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Subset</p></body></html>'
    )
    epub_subset = _make_minimal_epub_bytes(xhtml_subset)
    assert (
        zipguard.preflight_zip(epub_subset, label="epub", scan_xml_members=True)
        == "epub XML contains forbidden DOCTYPE or ENTITY declaration"
    )

    # Unknown non-HTML entity in XHTML is still rejected by the XML parser.
    xhtml_unknown_ent = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b"<!DOCTYPE html>\n"
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&bogusxxe;</p></body></html>'
    )
    epub_unknown = _make_minimal_epub_bytes(xhtml_unknown_ent)
    err_unknown = zipguard.preflight_zip(epub_unknown, label="epub", scan_xml_members=True)
    assert err_unknown is not None
    assert err_unknown.startswith("epub XML parse error:")


def test_epub_with_plain_doctype_and_nbsp_converts_and_entity_rejected() -> None:
    pytest.importorskip("markitdown")
    from voice_agent.rag.convert import convert_with_markitdown

    xhtml_ok = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b"<!DOCTYPE html>\n"
        b'<html xmlns="http://www.w3.org/1999/xhtml">'
        b"<head><title>Ch 1</title></head>"
        b"<body><h1>Pump Guide</h1><p>Check pressure&nbsp;daily.</p></body>"
        b"</html>"
    )
    res_ok = convert_with_markitdown(_make_minimal_epub_bytes(xhtml_ok), "epub")
    assert res_ok.status == "ok"
    assert "Pump Guide" in res_ok.text

    xhtml_entity = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<!DOCTYPE html [<!ENTITY xxe "secret">]>\n'
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&xxe;</p></body></html>'
    )
    res_bad = convert_with_markitdown(_make_minimal_epub_bytes(xhtml_entity), "epub")
    assert res_bad.status == "unreadable"
    assert res_bad.error == "epub XML contains forbidden DOCTYPE or ENTITY declaration"


def test_doctype_scan_is_linear_and_rejects_unclosed_before_next_tag() -> None:
    payload = b"<!DOCTYPE " * 200_000 + b">"
    start = time.perf_counter()
    rejected_plain = zipguard.has_forbidden_xml_decl(payload, allow_plain_doctype=True)
    elapsed = time.perf_counter() - start
    assert rejected_plain is True
    assert elapsed < 5.0

    # Multiple closed plain DOCTYPEs also stay linear and are accepted under plain mode.
    closed_many = b"<!DOCTYPE html> " * 50_000
    start_closed = time.perf_counter()
    accepted_many = zipguard.has_forbidden_xml_decl(closed_many, allow_plain_doctype=True)
    elapsed_closed = time.perf_counter() - start_closed
    assert accepted_many is False
    assert elapsed_closed < 5.0

    # Quoted '<' and '>' inside a single DOCTYPE remain valid; unclosed '<' before '>' rejects.
    assert zipguard._has_forbidden_doctype('<!DOCTYPE foo "a>b" "c<d" >') is False
    assert zipguard._has_forbidden_doctype("<!DOCTYPE html <p>hi</p>") is True
    assert zipguard._has_forbidden_doctype("<!DOCTYPE html><!DOCTYPE html>") is False


def test_svg_member_allows_plain_doctype_but_rejects_entity_and_internal_subset() -> None:
    svg_ok = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        b'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b'<rect width="10" height="10"/>'
        b"</svg>"
    )
    pptx_ok = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/slides/slide1.xml", b'<?xml version="1.0"?><sld/>'),
            ("ppt/media/image9.svg", svg_ok),
        ]
    )
    assert zipguard.preflight_zip(pptx_ok, label="pptx", scan_xml_members=True) is None

    svg_entity = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!ENTITY xxe "boom">\n'
        b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    )
    pptx_entity = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/image9.svg", svg_entity),
        ]
    )
    assert (
        zipguard.preflight_zip(pptx_entity, label="pptx", scan_xml_members=True)
        == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"
    )

    svg_param_entity = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!ENTITY % pe "boom">\n'
        b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    )
    pptx_param = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/image9.svg", svg_param_entity),
        ]
    )
    assert (
        zipguard.preflight_zip(pptx_param, label="pptx", scan_xml_members=True)
        == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"
    )

    svg_subset = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        b'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [<!ELEMENT svg ANY>]>\n'
        b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    )
    pptx_subset = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/image9.svg", svg_subset),
        ]
    )
    assert (
        zipguard.preflight_zip(pptx_subset, label="pptx", scan_xml_members=True)
        == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"
    )


def test_high_ratio_media_exempt_while_parsed_members_rejected() -> None:
    emf_payload = b"\x01\x00\x00\x00" + b"\x00" * 512_000
    raw_emf = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/image22.emf", emf_payload),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    with zipfile.ZipFile(io.BytesIO(raw_emf)) as zf:
        emf_info = zf.getinfo("ppt/media/image22.emf")
        assert emf_info.file_size / emf_info.compress_size >= 500

    assert zipguard.preflight_zip(raw_emf, label="pptx", scan_xml_members=True) is None
    assert zipguard.preflight_zip(raw_emf, label="pptx", scan_xml_members=False) is None

    raw_bin = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/embeddings/oleObject1.bin", b"\x00" * 512_000),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_bin, label="pptx", scan_xml_members=True) is None

    raw_doc_bomb = _make_zip(
        [("word/document.xml", b"A" * 50_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(raw_doc_bomb, label="docx", scan_xml_members=False)
        == "suspicious compression ratio in 'word/document.xml'"
    )

    for suffix in (
        ".xml",
        ".rels",
        ".opf",
        ".ncx",
        ".xhtml",
        ".html",
        ".htm",
        ".txt",
        ".csv",
        ".json",
        ".svg",
    ):
        member_name = f"parsed/bomb{suffix}"
        raw_parsed = _make_zip(
            [(member_name, b"A" * 50_000)],
            compression=zipfile.ZIP_DEFLATED,
        )
        assert (
            zipguard.preflight_zip(raw_parsed, label="pptx", scan_xml_members=False)
            == f"suspicious compression ratio in {member_name!r}"
        )


def test_high_ratio_media_still_bounded_by_member_and_total_caps() -> None:
    big_media = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/huge.emf", b"\x00" * 50_000),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(
            big_media,
            label="pptx",
            scan_xml_members=True,
            max_member_bytes=40_000,
            max_total_bytes=100_000,
        )
        == "pptx member 'ppt/media/huge.emf' exceeds 40000 uncompressed bytes"
    )

    many_media = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            *[(f"ppt/media/img_{i}.emf", b"\x01\x00\x00\x00" + b"\x00" * 30_000) for i in range(4)],
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(
            many_media,
            label="pptx",
            scan_xml_members=True,
            max_member_bytes=40_000,
            max_total_bytes=100_000,
        )
        == "pptx total uncompressed size exceeds 100000 bytes"
    )


def test_ratio_check_suffix_matching_is_case_insensitive() -> None:
    upper_media = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("PPT/MEDIA/IMAGE1.EMF", b"\x01\x00\x00\x00" + b"\x00" * 512_000),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(upper_media, label="pptx", scan_xml_members=True) is None

    for name in ("WORD/DOCUMENT.XML", "PPT/BOMB.TXT", "PPT/MEDIA/BAD.SVG"):
        upper_parsed = _make_zip([(name, b"A" * 50_000)], compression=zipfile.ZIP_DEFLATED)
        assert (
            zipguard.preflight_zip(upper_parsed, label="pptx", scan_xml_members=False)
            == f"suspicious compression ratio in {name!r}"
        )


def test_pptx_with_high_ratio_emf_passes_preflight_and_converts_ok() -> None:
    pptx = pytest.importorskip("pptx")
    from voice_agent.rag.convert import convert_with_direct

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    assert slide.shapes.title is not None
    slide.shapes.title.text = "Environmental Health Module 1"

    base_buf = io.BytesIO()
    prs.save(base_buf)
    base_buf.seek(0)

    out_buf = io.BytesIO()
    with (
        zipfile.ZipFile(base_buf, "r") as zin,
        zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            zout.writestr(item.filename, zin.read(item.filename))
        zout.writestr("ppt/media/image1.emf", b"\x00" * (1024 * 1024))

    pptx_bytes = out_buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as zf:
        emf_info = zf.getinfo("ppt/media/image1.emf")
        assert emf_info.file_size == 1024 * 1024
        assert emf_info.file_size / emf_info.compress_size > zipguard.MAX_ZIP_COMPRESSION_RATIO

    assert zipguard.preflight_zip(pptx_bytes, label="pptx", scan_xml_members=True) is None
    res = convert_with_direct(pptx_bytes, "pptx")
    assert res.status == "ok"
    assert "Environmental Health Module 1" in res.text


def test_high_ratio_media_single_and_aggregate_limits() -> None:
    assert zipguard.HIGH_RATIO_MEDIA_MAX_BYTES == 4 * 1024 * 1024
    assert zipguard.HIGH_RATIO_MEDIA_TOTAL_BYTES == 8 * 1024 * 1024

    # 1 MB-of-zeros EMF (and 1,160,132-byte real corpus size) passes.
    one_mb = 1024 * 1024
    raw_1mb = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/image1.emf", b"\x00" * one_mb),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_1mb, label="pptx", scan_xml_members=True) is None

    raw_corpus_emf = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/real.emf", b"\x00" * 1_160_132),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_corpus_emf, label="pptx", scan_xml_members=True) is None

    # 4 MB exact boundary passes; 5 MB zeros media member is rejected.
    raw_4mb = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/edge.emf", b"\x00" * (4 * 1024 * 1024)),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_4mb, label="pptx", scan_xml_members=True) is None

    raw_5mb = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/bomb.emf", b"\x00" * (5 * 1024 * 1024)),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(raw_5mb, label="pptx", scan_xml_members=True)
        == "suspicious compression ratio in 'ppt/media/bomb.emf'"
    )

    # Three 1 MB zeros media members (3 MB) pass; ten (10 MB) hit the 8 MB aggregate cap.
    raw_three = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            *[(f"ppt/media/img_{i}.emf", b"\x00" * one_mb) for i in range(3)],
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_three, label="pptx", scan_xml_members=True) is None

    raw_ten = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            *[(f"ppt/media/img_{i}.emf", b"\x00" * one_mb) for i in range(10)],
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(raw_ten, label="pptx", scan_xml_members=True)
        == f"high-ratio media members exceed {zipguard.HIGH_RATIO_MEDIA_TOTAL_BYTES} bytes"
    )

    # Exactly 8 MB across two 4 MB high-ratio members passes; adding 1 more KB fails aggregate.
    raw_8mb = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/a.emf", b"\x00" * (4 * 1024 * 1024)),
            ("ppt/media/b.emf", b"\x00" * (4 * 1024 * 1024)),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert zipguard.preflight_zip(raw_8mb, label="pptx", scan_xml_members=True) is None

    raw_over_8mb = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/a.emf", b"\x00" * (4 * 1024 * 1024)),
            ("ppt/media/b.emf", b"\x00" * (4 * 1024 * 1024)),
            ("ppt/media/c.emf", b"\x00" * 512_000),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert (
        zipguard.preflight_zip(raw_over_8mb, label="pptx", scan_xml_members=True)
        == f"high-ratio media members exceed {zipguard.HIGH_RATIO_MEDIA_TOTAL_BYTES} bytes"
    )


def test_low_ratio_large_media_unaffected_by_high_ratio_media_caps() -> None:
    import random

    rng = random.Random(0)
    ten_mb_random = rng.randbytes(10 * 1024 * 1024)
    raw_30mb_split = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/video1.bin", ten_mb_random),
            ("ppt/media/video2.bin", ten_mb_random),
            ("ppt/media/video3.bin", ten_mb_random),
        ],
        compression=zipfile.ZIP_STORED,
    )
    assert zipguard.preflight_zip(raw_30mb_split, label="pptx", scan_xml_members=True) is None

    thirty_mb_random = ten_mb_random * 3
    raw_30mb_single = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/media/video_single.bin", thirty_mb_random),
        ],
        compression=zipfile.ZIP_STORED,
    )
    assert (
        zipguard.preflight_zip(
            raw_30mb_single,
            label="pptx",
            scan_xml_members=True,
            max_member_bytes=32 * 1024 * 1024,
        )
        is None
    )
