"""Tests for direct HTML and EPUB readers (``readers.py``, ``convert.py``, ``docindex.py``)."""

from __future__ import annotations

import importlib.metadata
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

from voice_agent.rag import metrics, readers
from voice_agent.rag.convert import ConvertResult, convert_isolated, convert_with_direct
from voice_agent.rag.docindex import DocIndex


def _make_zip(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _make_epub(
    *,
    manifest_items: list[tuple[str, str, str]],
    spine_idrefs: list[str],
    files: dict[str, bytes],
    opf_path: str = "OEBPS/content.opf",
) -> bytes:
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles>"
        f'<rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/>'
        "</rootfiles>"
        "</container>"
    ).encode()
    manifest_xml = "\n".join(
        f'<item id="{item_id}" href="{href}" media-type="{media_type}"/>'
        for item_id, href, media_type in manifest_items
    )
    spine_xml = "\n".join(f'<itemref idref="{idref}"/>' for idref in spine_idrefs)
    opf_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Test EPUB</dc:title>"
        "</metadata>"
        f"<manifest>{manifest_xml}</manifest>"
        f"<spine>{spine_xml}</spine>"
        "</package>"
    ).encode()

    members: list[tuple[str, bytes]] = [
        ("mimetype", b"application/epub+zip"),
        ("META-INF/container.xml", container_xml),
        (opf_path, opf_xml),
    ]
    for path, content in files.items():
        members.append((path, content))
    return _make_zip(members)


def test_html_boilerplate_removed_and_main_kept_with_token_matching() -> None:
    pytest.importorskip("lxml")
    html = b"""<!DOCTYPE html>
<html>
<head>
  <title>Ignored Head Title</title>
  <style>.secret-css { color: red; }</style>
  <script>var secretJs = "DROP_JS";</script>
</head>
<body class="has-sidebar">
  <header>Site Header Banner</header>
  <nav>Home | Docs | Pricing</nav>
  <div id="cookie-consent-banner"><p>Accept cookies now</p></div>
  <div class="left-sidebar"><p>Sidebar promo</p></div>
  <div class="social-share-bar"><button>Tweet</button></div>
  <div class="breadcrumb"><a href="/a">Root</a> &gt; Page</div>
  <main>
    <p class="shareholder-note">Shareholder dividend is 42 percent.</p>
    <p class="navigation-guide">Navigation guide for field setup.</p>
    <script>console.log("INLINE_SCRIPT_DROP");</script>
    <noscript>Enable JS warning</noscript>
    <svg><text>SVG_DROP</text></svg>
    <form><input value="DROP_INPUT"/><button>Submit</button></form>
  </main>
  <aside>Related links</aside>
  <footer>Copyright 2026</footer>
</body>
</html>"""

    res = convert_with_direct(html, "html")
    assert res.status == "ok"
    assert res.converter == f"direct-lxml-{importlib.metadata.version('lxml')}"
    assert "Shareholder dividend is 42 percent." in res.text
    assert "Navigation guide for field setup." in res.text
    for dropped in (
        "secret-css",
        "DROP_JS",
        "INLINE_SCRIPT_DROP",
        "Site Header Banner",
        "Pricing",
        "Accept cookies now",
        "Sidebar promo",
        "Tweet",
        "Enable JS warning",
        "SVG_DROP",
        "DROP_INPUT",
        "Submit",
        "Related links",
        "Copyright 2026",
    ):
        assert dropped not in res.text


def test_html_main_and_article_root_selection() -> None:
    pytest.importorskip("lxml")
    html_main = b"""<html><body>
      <div class="intro">Outside main text</div>
      <main><h1>Primary Section</h1><p>Inside main content.</p></main>
    </body></html>"""
    res_main = convert_with_direct(html_main, "html")
    assert res_main.status == "ok"
    assert "Inside main content." in res_main.text
    assert "Outside main text" not in res_main.text

    html_article = b"""<html><body>
      <aside><article><p>Sidebar teaser should be dropped</p></article></aside>
      <div class="wrapper">Outside article text</div>
      <article><h2>Story Heading</h2><p>First article body.</p></article>
      <article><p>Second article ignored when first selected.</p></article>
    </body></html>"""
    res_article = convert_with_direct(html_article, "html")
    assert res_article.status == "ok"
    assert "Story Heading" in res_article.text
    assert "First article body." in res_article.text
    assert "Sidebar teaser should be dropped" not in res_article.text
    assert "Outside article text" not in res_article.text
    assert "Second article ignored" not in res_article.text

    html_role_main = b"""<html><body>
      <div>Before role main</div>
      <div role="main"><p>Role main body.</p></div>
    </body></html>"""
    res_role = convert_with_direct(html_role_main, "html")
    assert res_role.status == "ok"
    assert "Role main body." in res_role.text
    assert "Before role main" not in res_role.text


def test_html_markdown_elements_headings_lists_pre_blockquote_links_img_dl() -> None:
    pytest.importorskip("lxml")
    html = b"""<html><body><main>
      <h1>Main Title</h1>
      <h3>Sub Heading</h3>
      <p>Visit <a href="https://example.com/secret-url">Documentation</a><br>Second line.</p>
      <p><img src="a.png" alt="Sensor diagram"> and <img src="b.png" alt=""></p>
      <ul>
        <li>Alpha item</li>
        <li>Beta item</li>
      </ul>
      <ol>
        <li>Step one</li>
        <li>Step two</li>
      </ol>
      <blockquote><p>Quoted warning text.</p></blockquote>
      <pre><code>def ping():
    return 200</code></pre>
      <dl>
        <dt>Timeout</dt>
        <dd>30 seconds</dd>
      </dl>
    </main></body></html>"""

    raw_res = readers.read_html(html)
    assert raw_res.status == "ok"
    assert "# Main Title" in raw_res.text
    assert "### Sub Heading" in raw_res.text
    assert "Visit Documentation\nSecond line." in raw_res.text
    assert "https://example.com/secret-url" not in raw_res.text
    assert "Sensor diagram" in raw_res.text
    assert "- Alpha item\n- Beta item" in raw_res.text
    assert "1. Step one\n2. Step two" in raw_res.text
    assert "> Quoted warning text." in raw_res.text
    assert "```\ndef ping():\n    return 200\n```" in raw_res.text
    assert "Timeout: 30 seconds" in raw_res.text

    final_res = convert_with_direct(html, "html")
    assert final_res.status == "ok"
    assert "Timeout: 30 seconds" in final_res.text
    assert "https://example.com/secret-url" not in final_res.text


def test_html_tables_thead_colspan_rowspan_and_no_unnamed() -> None:
    pytest.importorskip("lxml")
    html = b"""<html><body><main>
      <table>
        <thead>
          <tr><th>Region</th><th colspan="2">2026 Output</th></tr>
        </thead>
        <tbody>
          <tr><td rowspan="2">North</td><td>Q1 | Plan</td><td>120</td></tr>
          <tr><td>Q2</td><td>150</td></tr>
        </tbody>
      </table>
      <table>
        <tr><th>Device</th><th>Status</th></tr>
        <tr><td>Pump A</td><td>OK</td></tr>
      </table>
      <table>
        <tr><td>Plain A</td><td>Plain B</td></tr>
        <tr><td>10</td><td>20</td></tr>
      </table>
    </main></body></html>"""

    raw_res = readers.read_html(html)
    assert raw_res.status == "ok"
    assert "| Region | 2026 Output | 2026 Output |" in raw_res.text
    assert "| North | Q1 \\| Plan | 120 |" in raw_res.text
    assert "| North | Q2 | 150 |" in raw_res.text
    assert "| Device | Status |" in raw_res.text
    assert "| Plain A | Plain B |" in raw_res.text
    assert "Unnamed" not in raw_res.text

    res = convert_with_direct(html, "html")
    assert res.status == "ok"
    assert "Unnamed" not in res.text
    assert "Region: North; 2026 Output: Q1 | Plan; 2026 Output: 120" in res.text
    assert "Region: North; 2026 Output: Q2; 2026 Output: 150" in res.text
    assert "Device: Pump A; Status: OK" in res.text
    assert "Plain A: 10; Plain B: 20" in res.text


def test_html_entities_and_non_utf8_latin1_meta_charset() -> None:
    pytest.importorskip("lxml")
    html = (
        '<!DOCTYPE html><html><head><meta charset="latin-1"></head>'
        "<body><p>Caf\xe9 &amp; r\xe9sum\xe9 &mdash; 5 &lt; 10.</p></body></html>"
    ).encode("latin-1")

    res = convert_with_direct(html, "html")
    assert res.status == "ok"
    assert "Café & résumé — 5 < 10." in res.text

    html_http_equiv = (
        '<html><head><meta http-equiv="Content-Type" '
        'content="text/html; charset=iso-8859-1"></head>'
        "<body><p>Na\xefve pi\xf1ata</p></body></html>"
    ).encode("latin-1")
    res_http = convert_with_direct(html_http_equiv, "html")
    assert res_http.status == "ok"
    assert "Naïve piñata" in res_http.text

    # BOM-prefixed UTF-8 and non-UTF-8 without meta declaration (charset-normalizer fallback)
    bom_html = b"\xef\xbb\xbf<html><body><p>UTF8 BOM ok</p></body></html>"
    assert "UTF8 BOM ok" in convert_with_direct(bom_html, "html").text

    no_meta_latin1 = (
        "<html><body><p>Ren\xe9 met \xc9lodie at the caf\xe9 in Z\xfcrich.</p></body></html>"
    ).encode("latin-1")
    res_cn = convert_with_direct(no_meta_latin1, "html")
    assert res_cn.status == "ok"
    assert "René" in res_cn.text


def test_html_deep_nesting_and_caps_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lxml")
    deep_html = (
        b"<html><body><h1>Top Header</h1>"
        + b"<div>" * 5000
        + b"<p>Deep Leaf Text</p>"
        + b"</div>" * 5000
        + b"</body></html>"
    )
    res_deep = convert_with_direct(deep_html, "html")
    assert res_deep.status == "ok"
    assert res_deep.fallback == "truncated"
    assert "# Top Header" in res_deep.text
    assert "Deep Leaf Text" not in res_deep.text

    # Node cap
    many_nodes = (
        b"<html><body>"
        + b"".join(f"<p>Node {i}</p>".encode("ascii") for i in range(50))
        + b"</body></html>"
    )
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_HTML_NODES", 15)
        res_nodes = convert_with_direct(many_nodes, "html")
    assert res_nodes.status == "ok"
    assert res_nodes.fallback == "truncated"
    assert "Node 0" in res_nodes.text
    assert "Node 49" not in res_nodes.text

    # 11 MB input cap -> too_large
    oversized = b"<html><body><p>" + (b"a" * (11 * 1024 * 1024)) + b"</p></body></html>"
    res_big = convert_with_direct(oversized, "html")
    assert res_big.status == "too_large"


def test_epub_spine_order_ignores_non_spine_and_guards_before_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lxml")
    epub_bytes = _make_epub(
        manifest_items=[
            ("ch3", "text/ch3.xhtml", "application/xhtml+xml"),
            ("ch1", "text/ch1.xhtml", "application/xhtml+xml"),
            ("ch2", "text/ch2.xhtml", "application/xhtml+xml"),
            ("extra", "text/unreferenced.xhtml", "application/xhtml+xml"),
            ("style", "styles/main.css", "text/css"),
        ],
        spine_idrefs=["ch1", "ch2", "ch3"],
        files={
            "OEBPS/text/ch1.xhtml": (
                b'<?xml version="1.0" encoding="utf-8"?>'
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                b"<nav>EPUB TOC</nav><h1>Chapter One</h1><p>First chapter text.</p>"
                b"</body></html>"
            ),
            "OEBPS/text/ch2.xhtml": (
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                b"<h1>Chapter Two</h1><p>Second chapter text.</p>"
                b"</body></html>"
            ),
            "OEBPS/text/ch3.xhtml": (
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                b"<h1>Chapter Three</h1><p>Third chapter text.</p>"
                b"</body></html>"
            ),
            "OEBPS/text/unreferenced.xhtml": (
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                b"<p>NOT_IN_SPINE_TEXT</p></body></html>"
            ),
            "OEBPS/styles/main.css": b"body { color: black; }",
        },
    )

    res = convert_with_direct(epub_bytes, "epub")
    assert res.status == "ok"
    assert res.converter == f"direct-lxml-{importlib.metadata.version('lxml')}"
    assert "EPUB TOC" not in res.text
    assert "NOT_IN_SPINE_TEXT" not in res.text
    p1 = res.text.index("Chapter One")
    p2 = res.text.index("Chapter Two")
    p3 = res.text.index("Chapter Three")
    assert p1 < p2 < p3

    # Guards must run before readers.read_epub on DOCTYPE+ENTITY
    called: list[str] = []

    def _forbid_epub(raw: bytes) -> ConvertResult:
        called.append("epub")
        raise AssertionError("read_epub must not be called on guard failure")

    evil_epub = _make_epub(
        manifest_items=[("ch1", "ch1.xhtml", "application/xhtml+xml")],
        spine_idrefs=["ch1"],
        files={
            "OEBPS/ch1.xhtml": (
                b'<?xml version="1.0"?>'
                b'<!DOCTYPE html [<!ENTITY xxe "SECRET">]>'
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&xxe;</p></body></html>'
            )
        },
    )
    with monkeypatch.context() as mp:
        mp.setattr(readers, "read_epub", _forbid_epub)
        res_evil = convert_with_direct(evil_epub, "epub")
    assert res_evil.status == "unreadable"
    assert "forbidden DOCTYPE or ENTITY" in (res_evil.error or "")
    assert not called

    # EPUB spine cap -> fallback="truncated", item byte cap -> too_large
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_EPUB_SPINE_ITEMS", 2)
        res_spine_cap = convert_with_direct(epub_bytes, "epub")
    assert res_spine_cap.status == "ok"
    assert res_spine_cap.fallback == "truncated"
    assert "Chapter One" in res_spine_cap.text
    assert "Chapter Two" in res_spine_cap.text
    assert "Chapter Three" not in res_spine_cap.text

    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_EPUB_ITEM_BYTES", 64)
        res_item_cap = convert_with_direct(epub_bytes, "epub")
    assert res_item_cap.status == "too_large"


def test_direct_html_isolated_matches_in_process() -> None:
    pytest.importorskip("lxml")
    html = (
        b"<html><body><main><h1>Pump Check</h1>"
        b"<p>Inspect pressure valve weekly.</p>"
        b"<table><tr><th>Metric</th><th>Target</th></tr><tr><td>PSI</td><td>45</td></tr></table>"
        b"</main></body></html>"
    )
    expected = convert_with_direct(html, "html")
    actual = convert_isolated(html, "html", engine="direct")
    assert expected.status == "ok"
    assert actual == expected


def test_direct_html_epub_missing_package_and_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lxml_html = pytest.importorskip("lxml.html")

    valid_html = b"<html><body><p>Hello world</p></body></html>"
    valid_epub = _make_epub(
        manifest_items=[("ch1", "ch1.xhtml", "application/xhtml+xml")],
        spine_idrefs=["ch1"],
        files={"OEBPS/ch1.xhtml": b"<html><body><p>Book chapter</p></body></html>"},
    )

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "lxml", None)
        mp.setitem(sys.modules, "lxml.html", None)
        assert readers.read_html(valid_html) == ConvertResult(
            text="", status="unsupported", error="lxml not installed"
        )
        assert readers.read_epub(valid_epub) == ConvertResult(
            text="", status="unsupported", error="lxml not installed"
        )
        assert convert_with_direct(valid_html, "html") == ConvertResult(
            text="", status="unsupported", error="lxml not installed"
        )
        assert convert_with_direct(valid_epub, "epub") == ConvertResult(
            text="", status="unsupported", error="lxml not installed"
        )

    (tmp_path / "page.html").write_bytes(valid_html)
    (tmp_path / "book.epub").write_bytes(valid_epub)
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "lxml", None)
        mp.setitem(sys.modules, "lxml.html", None)
        idx = DocIndex(
            tmp_path,
            converters={"html": "direct", "epub": "direct"},
            canonical_cache_dir=None,
        )
        idx.refresh()
        assert sorted(idx.failures()) == [
            ("book.epub", "unsupported", "lxml not installed"),
            ("page.html", "unsupported", "lxml not installed"),
        ]

    class _BoomHtmlError(RuntimeError):
        pass

    with monkeypatch.context() as mp:
        mp.setattr(
            lxml_html,
            "fromstring",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomHtmlError("SECRET_HTML_BODY")),
        )
        res_h = convert_with_direct(valid_html, "html")
        assert res_h.status == "unreadable"
        assert res_h.text == ""
        assert res_h.error == "html parse error: _BoomHtmlError"
        assert "SECRET_HTML_BODY" not in res_h.error

        res_e = convert_with_direct(valid_epub, "epub")
        assert res_e.status == "unreadable"
        assert res_e.text == ""
        assert res_e.error == "epub parse error: _BoomHtmlError"
        assert "SECRET_HTML_BODY" not in res_e.error


def test_docindex_direct_indexes_html_and_epub_with_cache(tmp_path: Path) -> None:
    pytest.importorskip("lxml")

    converters = metrics.resolve_converters_arg("html=direct,epub=direct")
    assert converters is not None
    assert converters["html"] == "direct"
    assert converters["epub"] == "direct"

    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "guide.html").write_bytes(
        b"<html><body><nav>Nav</nav>"
        b"<main><h1>Setup</h1><p>Plug sensor into port A.</p></main></body></html>"
    )
    (corpus / "notes.htm").write_bytes(
        b"<html><body><h1>Notes</h1><p>Inspect gasket seal monthly.</p></body></html>"
    )
    (corpus / "manual.epub").write_bytes(
        _make_epub(
            manifest_items=[
                ("ch1", "ch1.xhtml", "application/xhtml+xml"),
                ("ch2", "ch2.xhtml", "application/xhtml+xml"),
            ],
            spine_idrefs=["ch1", "ch2"],
            files={
                "OEBPS/ch1.xhtml": (
                    b"<html><body><h1>Calibration</h1><p>Use pH 7 buffer.</p></body></html>"
                ),
                "OEBPS/ch2.xhtml": (
                    b"<html><body><h1>Storage</h1><p>Keep probe capped.</p></body></html>"
                ),
            },
        )
    )

    idx = DocIndex(
        corpus,
        converters={"html": "direct", "epub": "direct"},
    )
    idx.refresh()
    assert idx.failures() == []
    sources = {c.source for c in idx.chunks}
    assert sources == {"guide.html", "manual.epub", "notes.htm"}
    assert any(
        c.source == "guide.html"
        and c.fmt == "html"
        and c.converter.startswith("direct-lxml-")
        and "Plug sensor into port A." in c.text
        and "Nav" not in c.text
        for c in idx.chunks
    )
    assert any(
        c.source == "manual.epub"
        and c.fmt == "epub"
        and c.converter.startswith("direct-lxml-")
        and "Use pH 7 buffer." in c.text
        for c in idx.chunks
    )

    # Warm index from canonical disk cache
    idx2 = DocIndex(
        corpus,
        converters={"html": "direct", "epub": "direct"},
    )
    idx2.refresh()
    assert idx2.failures() == []
    assert [c.id for c in idx2.chunks] == [c.id for c in idx.chunks]


def test_real_testset_html_smoke_if_available(tmp_path: Path) -> None:
    if os.environ.get("VOICE_AGENT_RAG_EXTERNAL_CORPUS") != "1":
        pytest.skip("private corpus smoke: set VOICE_AGENT_RAG_EXTERNAL_CORPUS=1 to run it")
    pytest.importorskip("lxml")
    testset_dir = Path.home() / "rag-testsets"
    if not testset_dir.is_dir():
        pytest.skip("~/rag-testsets not available in sandbox")
    html_files = sorted(testset_dir.rglob("*.html"))
    if not html_files:
        pytest.skip("no html files in ~/rag-testsets")
    failures: list[str] = []
    for path in html_files:
        raw = path.read_bytes()
        res = convert_with_direct(raw, "html")
        if res.status != "ok" or "Unnamed" in res.text:
            failures.append(f"{path.parent.name}/{path.name}: {res.status} {res.error}")
    assert not failures, "\n".join(failures)

    # Copy one real page into tmp_path and index via DocIndex (never touching ~/rag-testsets)
    copied = tmp_path / html_files[0].name
    copied.write_bytes(html_files[0].read_bytes())
    idx = DocIndex(tmp_path, converters={"html": "direct"}, canonical_cache_dir=None)
    idx.refresh()
    assert idx.failures() == []
    assert any(c.source == copied.name and c.fmt == "html" for c in idx.chunks)
