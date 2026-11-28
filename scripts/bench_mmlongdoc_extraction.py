"""MMLongBench-Doc text-extraction sufficiency check (ticket 275, read-only, free).

Answers the ticket's question without running the lane: for each in-scope
answerable dev question (split == dev, not out_of_scope_vision, not
unanswerable; the 82 headline rows), is the gold answer actually in the text
we extracted from the 25 corpus PDFs, and where it is not, why?

Checks (all read-only over out/mmlongdoc/corpus25/ and the eval/ manifests):
  a. Per-PDF health from the corpus DB: chunks, characters, pages with a
     chunk vs the manifest's page count (scanned_pages.per_pdf), pages with
     no chunk. Flags every PDF with 0 chunks or under 200 characters/page
     on average.
  b. Gold-in-text per row: the gold is normalised (strip '$', ',', '%' and
     the word "and"; thousands separators incl. the Indian grouping
     3,02,16,492; case; the "fiscal year"/"fiscal"/"FY" forms unified; List
     gold = EVERY item must appear) and checked (1) anywhere in the gold
     document's DB chunks; (2) on the gold's evidence_pages specifically
     (page metadata); (3) in an INDEPENDENT extraction of the gold PDF's
     gold pages (pypdf / pdfminer.six / pypdfium2 - whichever import in the
     venv; nothing is installed). Classes: FOUND-BOTH,
     FOUND-INDEPENDENT-ONLY (our reader loses it), FOUND-DB-ONLY,
     ABSENT-BOTH (derived/computed or figure-only), DOC-EMPTY.
  c. Table integrity for rows whose evidence source is Table: one free Qwen
     call per row decides whether the gold is a computed ratio/percentage
     and names the input line items; the row passes when every input label
     and its number sit in the SAME DB chunk (a label and its value split
     across chunks is a structure loss).
  d. The empty slide decks: the reader's own record (files.note) plus each
     independent extractor on pages 1-3 (first 300 chars reported) - text
     layer, image-only, or the reader errored.

Outputs (out/, never committed):
  out/mmlongdoc/extraction-dev.jsonl  one record per in-scope dev row
  out/mmlongdoc/extraction-dev.md     the class counts (overall / by
                                      evidence source / by answer_format),
                                      the per-PDF health table, the
                                      table-integrity result with verbatim
                                      chunk examples, the slide-deck finding

Free: the local Qwen relay only (one call per table row for c; use
--skip-table-llm to run with no relay at all). Nothing here modifies
ask_documents, agent_server, sufficiency, the readers or ingest. Run from
the repo root in a venv with the PDF libraries (the root .venv has pypdf,
pdfminer.six and pypdfium2):

  uv run python scripts/bench_mmlongdoc_extraction.py
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# the bench module lives next to this file (scripts/); run with scripts/ on
# sys.path (the ticket-273 convention) or import it from the test loader
import bench_mmlongdoc as bm  # noqa: E402

DEFAULT_OUT = bm.OUT_DIR / "extraction-dev.jsonl"
DEFAULT_MD = bm.OUT_DIR / "extraction-dev.md"
DIAGNOSE_PATH = bm.OUT_DIR / "diagnose-dev.jsonl"

THIN_CHARS_PER_PAGE = 200.0  # flag a PDF under this average (the ticket's threshold)
DECK_MIN_CHARS = bm.SCAN_MIN_CHARS  # 20: below this an extractor saw no usable text
N_EXAMPLES = 5
EXAMPLE_CHUNK_CHARS = 1200

CLASS_ORDER = (
    "FOUND-BOTH",
    "FOUND-INDEPENDENT-ONLY",
    "FOUND-DB-ONLY",
    "ABSENT-BOTH",
    "DOC-EMPTY",
)

# --- gold normalisation (pure) -------------------------------------------------
_FY_RE = re.compile(r"fiscal\s*(?:yrs?|year)?")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm_gold(s: Any) -> str:
    """The loose normalisation for the gold-in-text check, applied to the gold
    AND the text: casefold; drop the word "and" (a standalone word - "Band"
    keeps its); unify the "fiscal year"/"fiscal"/"FY" forms to "fy" stuck to
    the year; strip '$', ',' and '%' (so every thousands separator is dropped,
    incl. the Indian grouping 3,02,16,492 -> 30216492); pack all whitespace."""
    t = str(s or "").casefold()
    t = re.sub(r"\band\b", "", t)
    t = _FY_RE.sub("fy", t)
    for ch in ("$", ",", "%"):
        t = t.replace(ch, "")
    t = re.sub(r"\s+", "", t)
    return t


def norm_numbers(s: Any) -> set[float]:
    """The numbers in a string with every thousands separator dropped
    ('1,234.5' -> 1234.5, '3,02,16,492' -> 30216492.0)."""
    out: set[float] = set()
    for m in _NUM_RE.findall(str(s or "")):
        with contextlib.suppress(ValueError):
            out.add(float(m.replace(",", "")))
    return out


def _single_number(s: str) -> float | None:
    """The value of a string that is ONE pure number (after norm_gold), else
    None: '30216492.00' -> 30216492.0, '13.5million' -> None."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def gold_in_text(gold: Any, answer_format: str, text: str) -> tuple[bool, str]:
    """Does the normalised gold answer appear in ``text``? The normalised gold
    string, or - for Int/Float gold that is a pure number - exact numeric
    equality with one of the text's numbers (so 30216492.00 matches
    "3,02,16,492" but gold "3" never matches inside "23"). List gold: EVERY
    item must appear (string or number) - the ticket-274 matcher used a
    majority; this one is exact. Returns (found, how)."""
    t = norm_gold(text)
    g = norm_gold(gold)
    fmt = str(answer_format)
    if g:
        if fmt in ("Int", "Float") and _single_number(g) is not None:
            if _single_number(g) in norm_numbers(text):
                return True, "number"
            return False, ""
        if g in t:
            return True, "string"
    if fmt == "List" or (g.startswith("[") and g.endswith("]")):
        try:
            items = ast.literal_eval(str(gold))
        except (ValueError, SyntaxError):
            items = None
        if isinstance(items, (list, tuple)) and items:
            tnums = norm_numbers(text)
            found = 0
            for it in items:
                nit = norm_gold(it)
                if nit and nit in t:
                    found += 1
                    continue
                inums = norm_numbers(it)
                if inums and inums <= tnums:
                    found += 1
            if found == len(items):
                return True, f"list {found}/{len(items)}"
    return False, ""


def assign_class(n_chunks: int, found_db: bool, found_independent: bool) -> str:
    """The ticket's five classes: DOC-EMPTY (the gold document has no chunks
    in the DB at all), then FOUND-BOTH / FOUND-INDEPENDENT-ONLY (our reader
    loses it) / FOUND-DB-ONLY / ABSENT-BOTH (derived/computed or figure-only)."""
    if n_chunks == 0:
        return "DOC-EMPTY"
    if found_db and found_independent:
        return "FOUND-BOTH"
    if found_independent:
        return "FOUND-INDEPENDENT-ONLY"
    if found_db:
        return "FOUND-DB-ONLY"
    return "ABSENT-BOTH"


def classify_extraction(
    *,
    n_chunks: int,
    full_text: str,
    page_texts: list[str],
    gold: Any,
    answer_format: str,
    independent: dict[str, Any],
) -> dict[str, Any]:
    """The per-row gold-in-text check and class (pure: every signal is
    pre-computed and passed in, so the tests run with fakes). ``independent``
    maps extractor name -> {"found": bool, "how": str}. The class uses the
    DB-any check (1); the on-evidence-pages check (2) is reported alongside."""
    db_any, db_how = gold_in_text(gold, answer_format, full_text)
    if page_texts:
        db_pages, db_pages_how = gold_in_text(gold, answer_format, "\n".join(page_texts))
    else:
        db_pages, db_pages_how = False, ""
    found_ind = any(bool(v.get("found")) for v in independent.values())
    return {
        "class": assign_class(n_chunks, db_any, found_ind),
        "found_db_any": db_any,
        "found_db_how": db_how,
        "found_db_pages": db_pages,
        "found_db_pages_how": db_pages_how,
        "found_independent": found_ind,
    }


# --- per-PDF health (read-only over the corpus DB) ------------------------------
def pdf_health(db: Path) -> list[dict[str, Any]]:
    """Per-file health straight from the DB: chunk count, characters, and the
    sorted distinct pages that carry at least one chunk. One row per
    ``files`` row, ordered by rel_key."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        stats: dict[str, tuple[int, int]] = {}
        pages: dict[str, set[int]] = {}
        for rel_key, n, chars in con.execute(
            "SELECT rel_key, COUNT(*), COALESCE(SUM(LENGTH(text)), 0) "
            "FROM chunk_data GROUP BY rel_key"
        ):
            stats[str(rel_key)] = (int(n), int(chars))
        for rel_key, page in con.execute(
            "SELECT rel_key, page FROM chunk_data WHERE page IS NOT NULL"
        ):
            pages.setdefault(str(rel_key), set()).add(int(page))
        out = []
        for rel_key, status in con.execute("SELECT rel_key, status FROM files ORDER BY rel_key"):
            n, chars = stats.get(str(rel_key), (0, 0))
            out.append(
                {
                    "rel_key": str(rel_key),
                    "status": str(status),
                    "n_chunks": n,
                    "chars": chars,
                    "chunk_pages": sorted(pages.get(str(rel_key), set())),
                }
            )
        return out
    finally:
        con.close()


def merge_manifest_health(
    health: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach the manifest's page count (scanned_pages.per_pdf, the docs list
    as fallback) to each health row: average chars/page, the pages with no
    chunk, and the flag: 'zero-chunks' (0 chunks) or 'thin' (under
    THIN_CHARS_PER_PAGE characters/page on average)."""
    per: dict[str, dict[str, Any]] = {}
    for p in (manifest.get("scanned_pages") or {}).get("per_pdf") or []:
        per[str(p.get("path") or "")] = p
    docs: dict[str, dict[str, Any]] = {}
    for d in manifest.get("docs") or []:
        docs[str(d.get("doc_id") or Path(str(d.get("path") or "")).name)] = d
    out: list[dict[str, Any]] = []
    for h in health:
        name = Path(str(h["rel_key"])).name
        p = per.get(name, {})
        d = docs.get(name, {})
        pages = int(p.get("pages") or d.get("pages") or 0)
        scanned = int(p.get("scanned_pages") or 0)
        have = set(h["chunk_pages"])
        avg = round(h["chars"] / pages, 1) if pages else None
        if h["n_chunks"] == 0:
            flag = "zero-chunks"
        elif avg is not None and avg < THIN_CHARS_PER_PAGE:
            flag = "thin"
        else:
            flag = ""
        out.append(
            {
                **h,
                "manifest_pages": pages,
                "manifest_scanned_pages": scanned,
                "avg_chars_per_page": avg,
                "pages_with_chunk": len(have),
                "pages_with_no_chunk": [pg for pg in range(1, pages + 1) if pg not in have],
                "flag": flag,
            }
        )
    return out


# --- the independent extractors (whatever imports; nothing is installed) --------
def available_extractors() -> dict[str, Any]:
    """name -> callable(path: Path, pages: list[int]) -> str for the
    independent PDF extractors importable in this venv: pypdf, pdfminer.six
    and pdfium (via pypdfium2). Pages are 1-based; missing ones are skipped.
    A missing library is simply not in the result."""
    out: dict[str, Any] = {}
    try:
        from pypdf import PdfReader

        def _pypdf(path: Path, pages: list[int]) -> str:
            reader = PdfReader(str(path))
            parts = [
                (reader.pages[i].extract_text() or "")
                for i in (p - 1 for p in pages)
                if 0 <= i < len(reader.pages)
            ]
            return "\n".join(parts)

        out["pypdf"] = _pypdf
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text

        def _pdfminer(path: Path, pages: list[int]) -> str:
            return extract_text(str(path), page_numbers=[p - 1 for p in pages if p >= 1]) or ""

        out["pdfminer"] = _pdfminer
    except ImportError:
        pass
    try:
        import pypdfium2 as pdfium

        def _pdfium(path: Path, pages: list[int]) -> str:
            pdf = pdfium.PdfDocument(str(path))
            try:
                parts = [
                    (pdf[i].get_textpage().get_text_range())
                    for i in (p - 1 for p in pages)
                    if 0 <= i < len(pdf)
                ]
                return "\n".join(parts)
            finally:
                pdf.close()

        out["pypdfium2"] = _pdfium
    except ImportError:
        pass
    return out


def extract_pages(path: Path, extractors: dict[str, Any], pages: list[int]) -> dict[str, Any]:
    """Every independent extractor on the given (1-based) pages:
    name -> {"error", "chars", "first300", "text"}. Never raises: a failing
    extractor is recorded with its error."""
    out: dict[str, Any] = {}
    for name, fn in extractors.items():
        try:
            text = fn(path, list(pages))
            out[name] = {
                "error": None,
                "chars": len(text),
                "first300": text[:300],
                "text": text,
            }
        except Exception as exc:  # a corrupt page must not sink the row
            out[name] = {"error": str(exc), "chars": 0, "first300": "", "text": ""}
    return out


# --- table integrity (c) ----------------------------------------------------------
TABLE_INPUTS_SYSTEM = "You are an exact table-question analyst. Reply with JSON only."


def table_inputs_prompt(question: str, gold: Any) -> str:
    return (
        "A document-QA question whose evidence is a table. Decide whether the gold "
        "answer is a value stated directly in the table, or a value COMPUTED from "
        "the table (a ratio, percentage, turnover, growth, difference, sum or any "
        "other derived figure).\n"
        "If it is computed, list the input line items: the labels of the table rows "
        "or columns the question names or that the computation needs, as they would "
        "appear in the table, most specific first.\n"
        'Reply with JSON only: an object with "computed" (true or false), '
        '"inputs" (a list of input label strings, empty when not computed) and '
        '"why" (one short sentence).\n\n'
        f"Question: {question}\nGold answer: {gold}"
    )


def table_inputs(
    question: str, gold: Any, transport: Any, model: str
) -> tuple[bool, list[str], str, str]:
    """(computed, inputs, why, error) from one free Qwen call; never raises."""
    se = bm.load_sufficiency_eval()
    payload = se.relay_payload(
        model, TABLE_INPUTS_SYSTEM + se.JSON_ONLY_SUFFIX, table_inputs_prompt(question, gold)
    )
    try:
        data = transport.post(payload)
        text = data["choices"][0]["message"]["content"]
        reply = json.loads(text)
        inputs = [str(x).strip() for x in (reply.get("inputs") or []) if str(x).strip()]
        return bool(reply.get("computed")), inputs, str(reply.get("why") or ""), ""
    except Exception as exc:  # relay down, non-JSON, malformed
        return False, [], "", f"table-inputs call failed: {exc}"


def chunk_locality(chunks: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """The table-integrity check (pure). ``chunks`` is [{"text": ...}, ...];
    ``labels`` the input line items the question names. A label passes when it
    (normalised) appears in at least one chunk that ALSO carries a number -
    a label whose value sits in a different chunk (the table row split across
    a chunk boundary) is a structure loss. ``pass`` is True/False, or None
    when there is nothing to check."""
    per: list[dict[str, Any]] = []
    for label in labels:
        nl = norm_gold(label)
        matched: list[tuple[int, bool]] = []
        for i, c in enumerate(chunks):
            if nl and nl in norm_gold(c.get("text") or ""):
                matched.append((i, bool(_NUM_RE.search(str(c.get("text") or "")))))
        if not matched:
            per.append(
                {"label": label, "found": False, "chunk": None, "number_in_same_chunk": False}
            )
        else:
            per.append(
                {
                    "label": label,
                    "found": True,
                    "chunk": matched[0][0],
                    "chunks": [i for i, _ in matched],
                    "number_in_same_chunk": any(num for _, num in matched),
                }
            )
    if not labels:
        ok: bool | None = None
    else:
        ok = all(p["found"] and p["number_in_same_chunk"] for p in per)
    return {"n_labels": len(labels), "labels": per, "pass": ok}


# --- the empty slide decks (d) -----------------------------------------------------
def deck_verdict(extractor_results: dict[str, Any]) -> str:
    """The slide-deck verdict from the independent extractors' results:
    'image-only (no text layer)' when every extractor ran and returned no
    usable text, 'has a text layer (the ingest reader missed it)' when any
    extractor got text, 'reader errored' when the only failures were
    extractor errors."""
    if not extractor_results:
        return "no extractor available"
    texts = [v for v in extractor_results.values() if v.get("error") is None]
    errors = [v for v in extractor_results.values() if v.get("error") is not None]
    if any(v.get("chars", 0) >= DECK_MIN_CHARS for v in texts):
        return "has a text layer (the ingest reader missed it)"
    if errors and not texts:
        return "reader errored"
    return "image-only (no text layer)"


def slide_deck_findings(
    db: Path, corpus_dir: Path, extractors: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every corpus file with 0 chunks: the reader's own note (files.note)
    plus each independent extractor on pages 1-3 (first 300 chars) and the
    verdict. Read-only; a missing PDF is recorded, not an error."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        zero = con.execute(
            "SELECT f.rel_key, f.note FROM files f "
            "WHERE NOT EXISTS (SELECT 1 FROM chunk_data c WHERE c.rel_key = f.rel_key)"
        ).fetchall()
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for rel_key, note in sorted(zero):
        path = corpus_dir / str(rel_key)
        record: dict[str, Any] = {
            "pdf": str(rel_key),
            "files_note": str(note or ""),
            "pdf_local": path.is_file(),
            "extractors": {},
        }
        if path.is_file():
            for name, res in extract_pages(path, extractors, [1, 2, 3]).items():
                record["extractors"][name] = {
                    "error": res["error"],
                    "chars": res["chars"],
                    "first300": res["first300"],
                }
        record["verdict"] = deck_verdict(record["extractors"])
        out.append(record)
    return out


# --- data loading (read-only joins) -------------------------------------------------
def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a torn tail line: skipped
        out[str(rec.get("id") or "")] = rec
    return out


def in_scope_dev_rows() -> list[dict[str, Any]]:
    """The in-scope answerable dev rows: split=dev, not out_of_scope_vision,
    not unanswerable (the ticket-273 headline population, 82 rows)."""
    questions = bm.load_questions(bm.DEFAULT_QUESTIONS)
    return [
        r
        for r in questions
        if r["split"] == "dev"
        and not (r.get("flags") or {}).get("out_of_scope_vision")
        and not bm.is_unanswerable(r.get("answer"))
    ]


# --- the main check ------------------------------------------------------------------
def check_rows(
    rows: list[dict[str, Any]],
    *,
    db: Path,
    corpus_dir: Path,
    manifest: dict[str, Any],
    extractors: dict[str, Any],
    transport: Any | None,
    model: str,
    on_row: Any = None,
) -> list[dict[str, Any]]:
    """Run every check for every in-scope dev row and return the records
    (the jsonl contents). The DB text, the independent extractions and the
    table-inputs calls are all cached per document / page set."""
    health = {h["rel_key"]: h for h in merge_manifest_health(pdf_health(db), manifest)}
    doc_cache: dict[str, dict[str, Any]] = {}
    indep_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    table_llm_cache: dict[str, tuple[bool, list[str], str, str]] = {}

    def doc_of(doc_id: str) -> dict[str, Any]:
        if doc_id not in doc_cache:
            chunks: list[tuple[int, str]] = []
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                for page, text in con.execute(
                    "SELECT page, text FROM chunk_data WHERE rel_key = ? ORDER BY ord",
                    (doc_id,),
                ):
                    chunks.append((int(page), str(text)))
            finally:
                con.close()
            doc_cache[doc_id] = {
                "chunks": chunks,
                "n_chunks": len(chunks),
                "full": "\n".join(t for _, t in chunks),
                "by_page": {},
            }
            for page, text in chunks:
                doc_cache[doc_id]["by_page"].setdefault(page, []).append(text)
        return doc_cache[doc_id]

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        doc_id = str(row["doc_id"])
        gold = row.get("answer")
        fmt = str(row.get("answer_format") or "")
        ev = [p for p in (row.get("evidence_pages") or []) if isinstance(p, int) and p > 0]
        h = health.get(doc_id, {})
        manifest_pages = int(h.get("manifest_pages") or 0)
        ev_used = [p for p in ev if not manifest_pages or p <= manifest_pages]
        notes: list[str] = []
        if len(ev_used) != len(ev):
            dropped = sorted(set(ev) - set(ev_used))
            notes.append(f"evidence page(s) {dropped} outside the document (or invalid); skipped")

        doc = doc_of(doc_id)
        page_texts: list[str] = []
        for p in ev_used:
            page_texts.extend(doc["by_page"].get(p, []))

        # (3) the independent extraction of the gold pages (cached per doc+pages)
        key = (doc_id, tuple(sorted(set(ev_used))))
        if key not in indep_cache:
            pdf_path = corpus_dir / doc_id
            if pdf_path.is_file():
                indep_cache[key] = extract_pages(pdf_path, extractors, key[1])
            else:
                indep_cache[key] = {
                    name: {"error": "PDF not local", "chars": 0, "first300": "", "text": ""}
                    for name in extractors
                }
        ind = {}
        for name, res in indep_cache[key].items():
            found, how = gold_in_text(gold, fmt, res["text"])
            ind[name] = {
                "found": found,
                "how": how,
                "chars": res["chars"],
                "error": res["error"],
            }

        cls = classify_extraction(
            n_chunks=doc["n_chunks"],
            full_text=doc["full"],
            page_texts=page_texts,
            gold=gold,
            answer_format=fmt,
            independent=ind,
        )

        # (c) table integrity: only the rows whose evidence source is Table
        table: dict[str, Any] | None = None
        if bm.TABLE_SOURCE in (row.get("evidence_sources") or []):
            if transport is None:
                table = {
                    "computed": None,
                    "inputs": [],
                    "why": "",
                    "error": "no relay available (--skip-table-llm)",
                    "pass": None,
                    "labels": [],
                    "example_chunks": [],
                }
            else:
                if str(row["id"]) not in table_llm_cache:
                    table_llm_cache[str(row["id"])] = table_inputs(
                        str(row.get("question") or ""), gold, transport, model
                    )
                computed, inputs, why, err = table_llm_cache[str(row["id"])]
                if err:
                    table = {
                        "computed": None,
                        "inputs": [],
                        "why": "",
                        "error": err,
                        "pass": None,
                        "labels": [],
                        "example_chunks": [],
                    }
                elif not computed:
                    table = {
                        "computed": False,
                        "inputs": [],
                        "why": why,
                        "error": "",
                        "pass": None,
                        "labels": [],
                        "example_chunks": [],
                    }
                else:
                    loc = chunk_locality([{"text": t} for _, t in doc["chunks"]], inputs)
                    examples: list[dict[str, Any]] = []
                    for p in loc["labels"]:
                        if p["found"] and len(examples) < 2:
                            ci = p["chunk"]
                            page, text = doc["chunks"][ci]
                            examples.append(
                                {
                                    "page": page,
                                    "label": p["label"],
                                    "text": text[:EXAMPLE_CHUNK_CHARS],
                                }
                            )
                    table = {
                        "computed": True,
                        "inputs": inputs,
                        "why": why,
                        "error": "",
                        "pass": loc["pass"],
                        "labels": loc["labels"],
                        "example_chunks": examples,
                    }

        rec = {
            "id": str(row["id"]),
            "doc_id": doc_id,
            "question": str(row.get("question") or ""),
            "gold": gold,
            "answer_format": fmt,
            "evidence_pages": [p for p in (row.get("evidence_pages") or [])],
            "evidence_pages_used": ev_used,
            "evidence_sources": list(row.get("evidence_sources") or []),
            "stratum": str((row.get("flags") or {}).get("stratum") or ""),
            "doc_n_chunks": doc["n_chunks"],
            **cls,
            "independent": ind,
            "notes": notes,
        }
        if table is not None:
            rec["table"] = table
        records.append(rec)
        if on_row is not None:
            on_row(rec, i, len(rows))
    return records


# --- the Markdown report -------------------------------------------------------------
def _md_table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return lines


def _class_table(rows: list[dict[str, Any]], key: str) -> list[str]:
    keys = sorted({str(r.get(key) or "?") for r in rows})
    body = [
        [k, str(sum(1 for r in rows if str(r.get(key) or "?") == k))]
        + [
            str(sum(1 for r in rows if str(r.get(key) or "?") == k and r["class"] == c))
            for c in CLASS_ORDER
        ]
        for k in keys
    ]
    return _md_table([key, "n", *CLASS_ORDER], body)


def render_report(
    records: list[dict[str, Any]],
    health: list[dict[str, Any]],
    decks: list[dict[str, Any]],
    extractors_used: list[str],
    diagnose: dict[str, dict[str, Any]],
) -> str:
    L: list[str] = []
    L.append("# MMLongBench-Doc text-extraction check - dev (ticket 275)")
    L.append("")
    L.append(
        f"rows: {len(records)} in-scope answerable dev rows (split=dev, not "
        f"out_of_scope_vision, not unanswerable). Independent extractors used: "
        f"{', '.join(extractors_used) or 'NONE'}. Gold normalisation: strip '$', ',', "
        "'%' and the word 'and', every thousands separator (incl. Indian grouping), "
        "case, 'fiscal year'/'fiscal'/'FY' unified; List gold = every item must "
        "appear (numeric Int/Float gold also matches its number, separators dropped)."
    )
    L.append("")
    L.append(
        "Classes: FOUND-BOTH (the gold is in the DB text AND in an independent "
        "extraction of the gold pages); FOUND-INDEPENDENT-ONLY (an independent "
        "extractor finds it on the gold pages but our DB text does not - our "
        "reader loses it); FOUND-DB-ONLY (in the DB text but not in the "
        "independent extraction of the gold pages - e.g. the value sits on a "
        "non-evidence page); ABSENT-BOTH (nowhere - derived/computed or "
        "figure-only); DOC-EMPTY (the gold document has zero chunks in the DB)."
    )
    L.append("")
    L.append("## Class counts")
    L.append("")
    body = [[c, str(sum(1 for r in records if r["class"] == c))] for c in CLASS_ORDER]
    L += _md_table(["class", "n"], body)
    L.append("")
    L += ["## By evidence source", ""]
    L += _class_table(records, "stratum")
    L.append("")
    L += ["## By answer_format", ""]
    L += _class_table(records, "answer_format")
    L.append("")
    n_db = sum(1 for r in records if r["found_db_any"])
    n_pages = sum(1 for r in records if r["found_db_pages"])
    L.append(
        f"of the {len(records)} rows: {n_db} have the gold somewhere in the gold "
        f"document's DB text, {n_pages} have it on the gold's own evidence pages "
        "(page metadata). The gap is gold that lives in the document but not on "
        "the cited evidence pages."
    )
    L.append("")
    # --- per-PDF health
    L.append("## Per-PDF health (the corpus DB vs the manifest)")
    L.append("")
    body = [
        [
            h["rel_key"],
            str(h["manifest_pages"]),
            str(h["n_chunks"]),
            str(h["chars"]),
            str(h["avg_chars_per_page"]),
            f"{h['pages_with_chunk']}/{h['manifest_pages']}",
            str(len(h["pages_with_no_chunk"])),
            str(h["manifest_scanned_pages"]),
            h["flag"] or "-",
        ]
        for h in health
    ]
    L += _md_table(
        [
            "pdf",
            "pages (manifest)",
            "chunks",
            "chars",
            "avg chars/page",
            "pages w/ chunk",
            "pages w/o chunk",
            "scanned (manifest)",
            "flag",
        ],
        body,
    )
    L.append("")
    bad = [h for h in health if h["flag"]]
    if bad:
        L.append(
            "Flagged PDFs (0 chunks, or under "
            f"{THIN_CHARS_PER_PAGE:.0f} characters/page on average): "
            + "; ".join(
                f"`{h['rel_key']}` ({h['flag']}, {h['n_chunks']} chunks, "
                f"{h['avg_chars_per_page']} chars/page)"
                for h in bad
            )
        )
    else:
        L.append("No PDF is zero-chunk or thin.")
    L.append("")
    # --- table integrity
    trows = [r for r in records if r.get("table") is not None]
    if trows:
        checked = [r for r in trows if r["table"]["computed"] is True]
        not_computed = [r for r in trows if r["table"]["computed"] is False]
        unchecked = [r for r in trows if r["table"]["computed"] is None]
        passed = [r for r in checked if r["table"]["pass"] is True]
        failed = [r for r in checked if r["table"]["pass"] is False]
        L.append("## Table integrity (evidence source = Table)")
        L.append("")
        L.append(
            f"{len(trows)} table rows. One free Qwen call per row decides whether "
            f"the gold is a computed ratio/percentage and names the input line "
            f"items; the row passes when every input label and its number sit in "
            f"the SAME DB chunk. {len(checked)} rows have a computed gold and were "
            f"checked: {len(passed)} pass, {len(failed)} fail"
            + (f" (pass rate {len(passed) / len(checked):.0%})" if checked else "")
            + f"; {len(not_computed)} rows state the value directly (no check "
            "needed - their gold-in-text class covers them)"
            + (
                f"; {len(unchecked)} rows could not be checked (no relay/call failed)"
                if unchecked
                else ""
            )
            + "."
        )
        L.append("")
        for title, group in (("passing", passed), ("failing", failed)):
            L.append(f"### {title} rows - the extracted chunk text (verbatim)")
            L.append("")
            if not group:
                L.append("(none)")
                L.append("")
                continue
            for r in group[:N_EXAMPLES]:
                t = r["table"]
                L.append(f"- **{r['id']}** (gold {r['gold']!r}) — Q: {r['question'].strip()}")
                L.append(f"  - computed: {t['why']}; inputs: {', '.join(t['inputs']) or '(none)'}")
                for lab in t["labels"]:
                    state = (
                        "found w/ number in same chunk"
                        if lab["found"] and lab["number_in_same_chunk"]
                        else (
                            "found, NO number in the same chunk"
                            if lab["found"]
                            else "label not found in the DB text"
                        )
                    )
                    L.append(f"  - input {lab['label']!r}: {state}")
                for ex in t["example_chunks"]:
                    L.append(f"  - chunk on page {ex['page']} (matched {ex['label']!r}):")
                    L.append("    ```")
                    for line in ex["text"].splitlines():
                        L.append(f"    {line}")
                    L.append("    ```")
                L.append("")
    # --- the slide decks
    L.append("## The empty slide decks (0 chunks in the DB)")
    L.append("")
    if not decks:
        L.append("(none)")
    for d in decks:
        L.append(f"### `{d['pdf']}`")
        L.append("")
        L.append(f"- reader's record (files.note): {d['files_note'] or '(none)'}")
        for name, res in d["extractors"].items():
            if res["error"] is not None:
                L.append(f"- {name}: errored ({res['error']})")
            else:
                first = res["first300"].replace("\n", " ")
                L.append(
                    f"- {name}: {res['chars']} chars on pages 1-3: "
                    f"{first[:300]!r} (first 300 chars verbatim)"
                )
        L.append(f"- verdict: **{d['verdict']}**")
        L.append("")
    # --- cross-tab with ticket 274
    if diagnose:
        missed = [r for r in records if diagnose.get(r["id"], {}).get("status") != "accepted"]
        accepted = [r for r in records if diagnose.get(r["id"], {}).get("status") == "accepted"]
        L.append("## Cross-tab with the ticket-274 diagnosis (missed rows)")
        L.append("")
        L.append(
            f"of the {len(records)} rows, {len(missed)} got no answer from the "
            f"lane (ticket 274's no-answer rows) and {len(accepted)} were accepted:"
        )
        L.append("")
        body = []
        for c in CLASS_ORDER:
            n_m = sum(1 for r in missed if r["class"] == c)
            n_a = sum(1 for r in accepted if r["class"] == c)
            body.append([c, str(n_m), str(n_a)])
        L += _md_table(["class", "no answer (missed)", "accepted"], body)
        L.append("")
        r0 = sum(1 for r in missed if diagnose.get(r["id"], {}).get("self_contained") is False)
        L.append(
            f"of the {len(missed)} missed rows, {r0} are ticket-274 R0 (the "
            "question is not self-contained - no shared-corpus retriever can "
            "find the document, extraction or not)."
        )
        L.append("")
    # --- what could not be checked
    L.append("## What this check could not decide")
    L.append("")
    L.append(
        "- The independent extraction covers only the gold's EVIDENCE pages, not "
        "the whole PDF: a value a third extractor would find elsewhere in the "
        "document is still ABSENT-BOTH here."
    )
    L.append(
        "- Scanned pages (<20 chars/page) carry no text in the DB; whether the "
        "gold lives on one as an image cannot be told without OCR (ticket T5)."
    )
    L.append(
        "- The normalisation is fixed by the ticket (case, '$'/','/'%'/'and', "
        "thousands separators, fiscal/FY): other surface variants (dates written "
        "as 'July 2009' vs '2009-07', worded numbers, synonyms) read as ABSENT."
    )
    L.append(
        "- The table-integrity 'computed vs stated' decision and the input labels "
        "rest on one free Qwen call per row (printed per example above)."
    )
    L.append(
        "- FOUND-DB-ONLY rows are not proven reader losses: the gold may sit on a "
        "page the dataset did not cite as evidence, or the independent extractor "
        "may have lost it."
    )
    return "\n".join(L) + "\n"


# --- CLI --------------------------------------------------------------------------------
def parse_args(argv: Any = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="bench_mmlongdoc_extraction",
        description="MMLongBench-Doc text-extraction check (ticket 275, read-only, free)",
    )
    ap.add_argument("--db", default=str(bm.DEFAULT_DB), help="corpus DB path")
    ap.add_argument("--corpus-dir", default=str(bm.DEFAULT_CORPUS_DIR))
    ap.add_argument("--manifest", default=str(bm.DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="extraction jsonl")
    ap.add_argument("--md", default=str(DEFAULT_MD), help="Markdown report")
    ap.add_argument(
        "--skip-table-llm",
        action="store_true",
        help="no relay at all: the table-integrity computed/stated call is skipped",
    )
    ap.add_argument("--base-url", default=bm.base_url_default(), help="the Qwen relay base URL")
    ap.add_argument("--model", default=bm.DEFAULT_MODEL)
    return ap.parse_args(argv)


def main(argv: Any = None) -> int:
    args = parse_args(argv)
    db = Path(args.db)
    corpus_dir = Path(args.corpus_dir)
    if not db.is_file():
        print(f"corpus DB not found: {db} (run ticket 273's ingest first)", file=sys.stderr)
        return 2
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = in_scope_dev_rows()
    if not rows:
        print("no in-scope dev rows in the questions file", file=sys.stderr)
        return 2
    extractors = available_extractors()
    print(
        f"independent extractors available in this venv: {sorted(extractors) or 'none'}",
        file=sys.stderr,
    )
    transport: Any = None
    if not args.skip_table_llm:
        try:
            se = bm.load_sufficiency_eval()
            transport = se.Relay(args.base_url, bm.read_key(), args.model)
            bm.relay_preflight(transport, args.model)
        except Exception as exc:  # relay down: degrade, the DB checks are relay-free
            print(
                f"warning: relay unavailable ({exc}); skipping the table-inputs call",
                file=sys.stderr,
            )
            transport = None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:

        def on_row(rec: dict[str, Any], i: int, n: int) -> None:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"  [{i}/{n}] {rec['id']} {rec['class']}",
                file=sys.stderr,
            )

        records = check_rows(
            rows,
            db=db,
            corpus_dir=corpus_dir,
            manifest=manifest,
            extractors=extractors,
            transport=transport,
            model=args.model,
            on_row=on_row,
        )
    diagnose = _load_jsonl_by_id(DIAGNOSE_PATH)
    health = merge_manifest_health(pdf_health(db), manifest)
    decks = slide_deck_findings(db, corpus_dir, extractors)
    md = render_report(records, health, decks, sorted(extractors), diagnose)
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    counts = Counter(r["class"] for r in records)
    print(
        "classes: "
        + "  ".join(f"{c}={counts.get(c, 0)}" for c in CLASS_ORDER)
        + f"  (total {len(records)})",
        file=sys.stderr,
    )
    flagged = [h["rel_key"] for h in health if h["flag"]]
    print(
        "flagged PDFs: " + (", ".join(flagged) if flagged else "none"),
        file=sys.stderr,
    )
    print(md)
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
