"""MMLongBench-Doc no-answer diagnosis (ticket 274, read-only).

For each in-scope answerable dev row (not out_of_scope_vision, not
unanswerable; 82 rows) this re-runs, exactly as the ticket-273 runner does,
the retrieval step (`context_fn`) and the lane call (`ask_documents`,
docs-only, local Qwen relay) - the lane is re-asked for the rows the original
run did not accept (~1 s each), so `DocResult.rejected` can be read - and
classifies why the lane gave no answer:

  R0  the question is not self-contained: it only makes sense with the document
      in hand, so no retriever over the shared 25-PDF corpus can find the
      document (the official task gives one PDF per question). Decided by a
      Qwen yes/no judge on the question text ALONE (and printed, hand-readable).
  R1  retrieval miss: the gold evidence page of the gold document is NOT in the
      passages the lane was given - checked by page number in the passage
      metadata, and also by whether the gold answer string / its normalised
      number appears in any passage of the gold document (both missing = R1).
  R2  the evidence page is in the passages but the gold answer string (or its
      normalised number) does not appear in the extracted text of the gold
      document's passages (scanned page, chart/table flattened, number split).
  R3  the evidence page is in the passages AND the gold answer (string or
      normalised number, or a majority of the List items) IS in the gold
      document's passage text, yet the lane returned none / check_failed
      (strict acceptance, quote check, schema; the failed check is recorded
      from the re-asked `DocResult.rejected`).
  R4  other / cannot tell (the lane call errored, or no signal applies).

Outputs (out/, never committed):
  out/mmlongdoc/diagnose-dev.jsonl  one record per in-scope row
  out/mmlongdoc/diagnose-dev.md     the class table, the R0 count over all 82
                                    rows, the scanned share of R1's gold pages,
                                    5 example rows per class
  out/mmlongdoc/diagnose-scanned-pages.json  per-page recomputation of the
                                    manifest's <20 chars/page scanned heuristic

Free: the local Qwen relay only (judge + lane re-ask). Nothing here modifies
ask_documents, agent_server, sufficiency, readers or ingest. Run from the
pipeline project directory (the ticket-273 convention):

  cd pipeline
  uv run --with pypdf python ../scripts/bench_mmlongdoc_diagnose.py
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# the bench module lives next to this file (scripts/); run with scripts/ on
# sys.path (the ticket-273 convention) or import it from the test loader
import bench_mmlongdoc as bm  # noqa: E402

DEFAULT_OUT = bm.OUT_DIR / "diagnose-dev.jsonl"
DEFAULT_MD = bm.OUT_DIR / "diagnose-dev.md"
SCANNED_CACHE = bm.OUT_DIR / "diagnose-scanned-pages.json"
N_EXAMPLES = 5

# --- R0: the self-containedness judge -----------------------------------------
JUDGE_SYSTEM = "You are a strict judge deciding whether a document-QA question is self-contained."


def judge_prompt(question: str) -> str:
    """The R0 judge's user message: the question text ALONE - no document, no
    gold answer, no file name. The judge must decide from the question text
    only, exactly as a shared-corpus retriever would see it."""
    return (
        "The official version of this benchmark gives the model exactly ONE PDF "
        "per question. Judge the question text ALONE, with no document in hand.\n"
        "A question is SELF-CONTAINED if its text identifies the specific document "
        "it is about (by title, author, institution, place, date or other "
        "distinctive subject) well enough that a catalog or search engine could "
        "find that document without ever having seen the document.\n"
        "A question that only makes sense with one particular document in hand "
        "(for example 'Which individual led this program?', 'What is the title of "
        "this document?', 'How many pages does the report have?') is NOT "
        "self-contained.\n"
        'Reply with JSON only: an object with the keys "self_contained" (true or '
        'false) and "why" (one short sentence).\n\nQuestion:\n' + question.strip()
    )


def judge_self_contained(transport: Any, model: str, question: str) -> tuple[bool | None, str]:
    """One free Qwen call on the question text alone. (True/False, why);
    (None, error) when the relay call or its JSON fails - never an exception,
    so one dead judge call cannot sink the whole diagnosis."""
    se = bm.load_sufficiency_eval()
    payload = se.relay_payload(model, JUDGE_SYSTEM + se.JSON_ONLY_SUFFIX, judge_prompt(question))
    try:
        data = transport.post(payload)
        text = data["choices"][0]["message"]["content"]
        reply = json.loads(text)
        flag = bool(reply.get("self_contained"))
        return flag, str(reply.get("why") or "")
    except Exception as exc:  # relay down, non-JSON, malformed
        return None, f"judge error: {exc}"


# --- R1/R2/R3 signals over the passages the lane was given ----------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm_text(s: Any) -> str:
    """Casefold + whitespace-collapsed text: the loose substring match used for
    the gold-answer-in-text check (R2/R3)."""
    return re.sub(r"\s+", " ", str(s).casefold()).strip()


def norm_numbers(s: Any) -> set[float]:
    """The numbers in a string, normalised (commas dropped): '1,234.5' -> 1234.5."""
    out: set[float] = set()
    for m in _NUM_RE.findall(str(s)):
        with contextlib.suppress(ValueError):
            out.add(float(m.replace(",", "")))
    return out


def gold_answer_in_text(gold: Any, answer_format: str, text: str) -> tuple[bool, str]:
    """Does the gold answer appear in ``text``? The gold answer string (case/
    whitespace-insensitive), or its normalised number for numeric answers, or
    a majority of the List items for List answers. Returns (found, how)."""
    t = norm_text(text)
    g = norm_text(gold)
    if g and g in t:
        return True, "string"
    gn = norm_numbers(gold)
    if gn and gn & norm_numbers(text):
        return True, "number"
    if str(answer_format) == "List":
        try:
            items = ast.literal_eval(str(gold))
        except (ValueError, SyntaxError):
            items = []
        if isinstance(items, (list, tuple)) and items:
            found = sum(1 for it in items if norm_text(it) and norm_text(it) in t)
            if found * 2 >= len(items):
                return True, f"list {found}/{len(items)}"
    return False, ""


def _page_int(p: Any) -> int | None:
    try:
        v = int(p)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def gold_doc_analysis(
    doc_id: str, evidence_pages: Any, gold: Any, answer_format: str, passages: Any
) -> dict[str, Any]:
    """The retrieval signals for the gold document from the raw ``context_fn``
    output (all passages the lane was given, in rank order): was the document
    retrieved at all, is an evidence page in the passage metadata, is the whole
    file in context, and does the gold answer appear in the gold document's
    passage text."""
    gold_passages = [p for p in passages if Path(str(p.get("source") or "")).name == str(doc_id)]
    pages_meta = sorted(
        {v for v in (_page_int(p.get("page")) for p in gold_passages) if v is not None}
    )
    whole = any(str(p.get("id") or "").endswith("#whole") for p in gold_passages)
    ev = [v for v in (_page_int(x) for x in (evidence_pages or [])) if v is not None]
    gold_text = "\n".join(str(p.get("text") or "") for p in gold_passages)
    answer_in_text, how = gold_answer_in_text(gold, answer_format, gold_text)
    return {
        "doc_retrieved": bool(gold_passages),
        "retrieved_pages_of_gold_doc": pages_meta,
        "whole_file": whole,
        "page_in_meta": whole or any(p in pages_meta for p in ev),
        "n_gold_passages": len(gold_passages),
        "answer_in_text": answer_in_text,
        "answer_how": how,
    }


# --- the classifier (pure: fake context + fake lane in the unit test) -----------
def classify_row(
    *,
    doc_id: str,
    evidence_pages: Any,
    gold: Any,
    answer_format: str,
    passages: Any,
    self_contained: bool | None,
    status: str,
    rejected: Any = None,
    status_reask: str | None = None,
) -> dict[str, Any]:
    """Classify one no-answer row R0-R4 from (fakeable) signals only.

    ``passages`` is the raw ``context_fn`` output the lane was given;
    ``status`` is the lane status being diagnosed (the original run's);
    ``rejected`` is the re-asked ``DocResult.rejected`` (check_failed rows);
    ``status_reask`` is the fresh lane status when the lane was re-asked.
    Accepted rows are class ``accepted`` (the class table is over the no-answer
    rows). Priority for the rest: R0 (not self-contained - no shared-corpus
    retriever can recover) > R1 (evidence page not in the passages: no page by
    metadata AND the gold answer not in any gold-doc passage; the gold doc
    missing entirely is R1 too) > R2 (page in the passages, answer not in the
    text) > R3 (answer in the gold-doc passage text, lane declined) > R4
    (cannot tell)."""
    a = gold_doc_analysis(doc_id, evidence_pages, gold, answer_format, passages)
    notes: list[str] = []
    if status_reask and status_reask != status:
        notes.append(f"re-ask status {status_reask} (original {status})")
    if self_contained is None:
        notes.append("the R0 judge call failed; the class rests on retrieval signals")
    if str(status) == "accepted":
        return {
            **a,
            "class": "accepted",
            "note": "the lane accepted this row",
            "notes": notes,
            "rejected_failed": "",
        }
    if str(status) == "error":
        cls, note = "R4", "the lane call errored (relay failure); cannot tell"
    elif self_contained is False:
        cls, note = "R0", "the question is not self-contained (judge, question text alone)"
    elif not a["doc_retrieved"]:
        cls, note = "R1", "the gold document is not in the retrieved passages at all"
    elif a["page_in_meta"] and not a["answer_in_text"]:
        cls, note = (
            "R2",
            (
                "an evidence page is in the passages but the gold answer (string / "
                "normalised number) is not in the gold document's passage text"
            ),
        )
    elif a["answer_in_text"]:
        cls, note = (
            "R3",
            ("the gold answer is in the gold document's passage text but the lane declined"),
        )
        if not a["page_in_meta"]:
            notes.append(
                "the evidence page is not in the passage metadata, but the gold "
                "answer text IS in a gold-doc passage (grep/whole lines)"
            )
    else:
        cls, note = (
            "R1",
            (
                "no evidence page in the passage metadata and the gold answer not in "
                "any gold-doc passage (the gold doc was retrieved without its "
                "evidence)"
            ),
        )
    return {
        "class": cls,
        "note": note,
        "notes": notes,
        "doc_retrieved": a["doc_retrieved"],
        "retrieved_pages_of_gold_doc": a["retrieved_pages_of_gold_doc"],
        "whole_file": a["whole_file"],
        "page_in_meta": a["page_in_meta"],
        "n_gold_passages": a["n_gold_passages"],
        "answer_in_text": a["answer_in_text"],
        "answer_how": a["answer_how"],
        "rejected_failed": str((rejected or {}).get("failed") or "") if rejected else "",
    }


# --- scanned pages (per-page recomputation of the manifest heuristic) -----------
def zero_chunk_docs(db: Path) -> list[str]:
    """Corpus files with status 'ok' but ZERO chunks in the DB: the reader
    extracted no text at all (e.g. image-only slide decks), so no text lane can
    ever answer a question about them, no matter how good the retrieval."""
    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT rel_key FROM files WHERE status = 'ok' "
            "AND NOT EXISTS (SELECT 1 FROM chunk_data c WHERE c.rel_key = files.rel_key)"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return sorted(str(r[0]) for r in rows)


def answer_in_gold_doc_db(db: Path, doc_id: str, gold: Any, answer_format: str) -> bool | None:
    """Does the gold answer appear ANYWHERE in the gold document's full DB text
    (every chunk, not just the retrieved passages)? None when the doc is not in
    the DB. This separates R2's 'extraction failed on the evidence page' from
    'the gold is a derived/calculated value that never appears in the document'
    (the R2 rows verified against the raw PDFs are the latter)."""
    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT text FROM chunk_data WHERE rel_key = ?", (str(doc_id),)
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not rows:
        return None
    full = "\n".join(str(t) for (t,) in rows)
    return gold_answer_in_text(gold, answer_format, full)[0]


def scanned_pages_map(data_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """doc_id -> {"pages_total": n, "scanned": [1-based page numbers]} for the
    manifest's PDFs: the ticket-273 <20 chars/page heuristic (pypdf) recomputed
    PER PAGE (the manifest records only the per-PDF count). Cached in
    out/mmlongdoc/ (never committed); a cache entry is trusted only when the
    file it describes still exists."""
    if SCANNED_CACHE.is_file():
        try:
            cached = json.loads(SCANNED_CACHE.read_text(encoding="utf-8"))
            if cached.get("min_chars_per_page") == bm.SCAN_MIN_CHARS:
                return cached["per_pdf"]
        except (ValueError, KeyError):
            pass  # torn cache: recompute
    out: dict[str, Any] = {}
    for p in manifest.get("pdfs") or []:
        path = data_dir / str(p.get("path") or "")
        doc_id = str(p.get("doc_id") or Path(str(p.get("path") or "")).name)
        if not path.is_file():
            out[doc_id] = {"pages_total": 0, "scanned": []}
            continue
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            scanned: list[int] = []
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if len(text.strip()) < bm.SCAN_MIN_CHARS:
                    scanned.append(i)
            out[doc_id] = {"pages_total": len(reader.pages), "scanned": scanned}
        except Exception as exc:
            print(f"warning: cannot scan {path.name} ({exc})", file=sys.stderr)
            out[doc_id] = {"pages_total": 0, "scanned": []}
    SCANNED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCANNED_CACHE.write_text(
        json.dumps({"min_chars_per_page": bm.SCAN_MIN_CHARS, "per_pdf": out}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


# --- data loading ----------------------------------------------------------------
def load_run_records(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise RuntimeError(f"run file not found: {path} (run ticket 273's `run` first)")
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise RuntimeError(f"{path}:{i + 1}: not a JSON line ({exc})") from None
        out[str(rec["id"])] = rec
    return out


def in_scope_dev_rows() -> list[dict[str, Any]]:
    """The 82 in-scope answerable dev rows: split=dev, not out_of_scope_vision,
    not unanswerable (the ticket-273 headline population)."""
    questions = bm.load_questions(bm.DEFAULT_QUESTIONS)
    return [
        r
        for r in questions
        if r["split"] == "dev"
        and not (r.get("flags") or {}).get("out_of_scope_vision")
        and not bm.is_unanswerable(r.get("answer"))
    ]


# --- the diagnose command ----------------------------------------------------------
def make_lane(
    se: Any,
    index: Any,
    file_names: list[str],
    cards: dict[str, dict[str, Any]],
    transport: Any,
    model: str,
    now: Any,
    budget_s: float,
) -> Any:
    """The runner's lane, verbatim (ticket 273's cmd_run): the same context_fn,
    the same Qwen doc request, the same ask_documents docs-only wiring."""
    from voice_pipeline.ask_documents import ask_documents
    from voice_pipeline.doc_answer import passages_with_meta

    async def context_fn(question: str) -> list[dict[str, Any]]:
        return passages_with_meta(await asyncio.to_thread(index.file_context, question))

    failures: list[str] = []
    doc_request = bm.make_doc_request(transport, model, se, failures)

    async def ask(question: str, row: Any) -> Any:
        return await ask_documents(
            question,
            context=context_fn,
            request=doc_request,
            web=None,
            catalog=None,
            file_names=file_names,
            card=lambda k: cards.get(str(k)),
            file_has=None,
            now=now,
            budget_s=budget_s,
            allow_fallback=False,
        )

    return ask


def top_files(passages: Any, n: int = 3) -> list[str]:
    """The top files retrieved, in the context_fn's rank order (distinct
    sources, first n)."""
    out: list[str] = []
    for p in passages:
        s = str(p.get("source") or "")
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out


async def diagnose_rows(args: argparse.Namespace) -> int:
    se = bm.load_sufficiency_eval()
    now = se.manila_now(args.today)
    if now is None:
        print(f"--today must be a YYYY-MM-DD date (got {args.today!r})", file=sys.stderr)
        return 2
    rows = in_scope_dev_rows()
    run_path = bm.OUT_DIR / f"run-{args.tag}.jsonl"
    run = load_run_records(run_path)
    score_path = bm.OUT_DIR / f"score-{args.tag}.json"
    score: dict[str, dict[str, Any]] = {}
    if score_path.is_file():
        report = json.loads(score_path.read_text(encoding="utf-8"))
        score = {str(r["id"]): r for r in (report.get("rows") or [])}

    db = Path(args.db)
    corpus_dir = Path(args.corpus_dir)
    from voice_pipeline.doc_answer import passages_with_meta

    from voice_agent.rag.dbindex import DbIndex

    index = DbIndex(db, raw_root=corpus_dir)
    file_names = se.load_file_names(db)
    cards = se.load_cards(db)
    transport = se.Relay(args.base_url, bm.read_key(), args.model)
    print("relay preflight (one tiny free call)...", file=sys.stderr)
    bm.relay_preflight(transport, args.model)
    ask = make_lane(se, index, file_names, cards, transport, args.model, now, args.budget_s)

    out_path = Path(args.out)
    done = {i for i in bm.read_done_ids(out_path)}
    print(
        f"diagnosing {len(rows)} in-scope dev rows ({len(done)} already diagnosed in "
        f"{out_path}) -> {out_path}",
        file=sys.stderr,
    )
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_new = 0
    doc_db_cache: dict[tuple[str, str, str], bool | None] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows, start=1):
            qid = str(row["id"])
            if qid in done:
                continue
            question = str(row.get("question") or "")
            # the R0 judge: the question text alone (all in-scope rows)
            self_contained, why = await asyncio.to_thread(
                judge_self_contained, transport, args.model, question
            )
            # the retrieval step, exactly as the runner calls it
            passages = passages_with_meta(await asyncio.to_thread(index.file_context, question))
            orig = run.get(qid) or {}
            status = str(orig.get("status") or "")
            # the lane re-ask (rows the original run did not accept; ~1 s each)
            status_reask: str | None = None
            rejected: Any = None
            spoken_reask: str | None = None
            if status != "accepted":
                t1 = time.perf_counter()
                res = await ask(question, row)
                status_reask = str(getattr(res, "status", "error"))
                rejected = getattr(res, "rejected", None)
                sp = getattr(res, "spoken", None)
                spoken_reask = (str(sp).strip() or None) if sp else None
                print(
                    f"  [{i}/{len(rows)}] {qid} judge={self_contained} "
                    f"orig={status} reask={status_reask} "
                    f"{time.perf_counter() - t1:.1f}s",
                    file=sys.stderr,
                )
            cls = classify_row(
                doc_id=str(row["doc_id"]),
                evidence_pages=row.get("evidence_pages") or [],
                gold=row.get("answer"),
                answer_format=str(row.get("answer_format") or ""),
                passages=passages,
                self_contained=self_contained,
                status=status,
                rejected=rejected,
                status_reask=status_reask,
            )
            sc = score.get(qid) or {}
            s = float(sc["score"]) if status == "accepted" and "score" in sc else None
            doc_id = str(row["doc_id"])
            fmt = str(row.get("answer_format") or "")
            db_key = (doc_id, str(row.get("answer")), fmt)
            if db_key not in doc_db_cache:
                doc_db_cache[db_key] = answer_in_gold_doc_db(db, doc_id, row.get("answer"), fmt)
            rec = {
                "id": qid,
                "question": question,
                "doc_id": str(row["doc_id"]),
                "evidence_pages": [int(x) for x in (row.get("evidence_pages") or [])],
                "evidence_sources": list(row.get("evidence_sources") or []),
                "stratum": str((row.get("flags") or {}).get("stratum") or ""),
                "answer_format": str(row.get("answer_format") or ""),
                "gold": row.get("answer"),
                "status": status,
                "status_reask": status_reask,
                "self_contained": self_contained,
                "self_contained_why": why,
                "class": cls["class"],
                "class_note": cls["note"],
                "notes": cls["notes"],
                "doc_retrieved": cls["doc_retrieved"],
                "retrieved_pages_of_gold_doc": cls["retrieved_pages_of_gold_doc"],
                "whole_file": cls["whole_file"],
                "page_in_meta": cls["page_in_meta"],
                "n_gold_passages": cls["n_gold_passages"],
                "gold_answer_in_gold_passages": "Y" if cls["answer_in_text"] else "N",
                "answer_how": cls["answer_how"],
                "answer_in_doc_text": "Y"
                if doc_db_cache[db_key]
                else ("N" if doc_db_cache[db_key] is not None else None),
                "top_files_retrieved": top_files(passages),
                "rejected": dict(rejected) if isinstance(rejected, dict) else None,
                "rejected_failed": cls["rejected_failed"],
                "spoken_reask": spoken_reask,
                "score": s,
                "correct": (s == 1.0) if s is not None else None,
                "partial": (0.0 < s < 1.0) if s is not None else None,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            n_new += 1
    index.close()
    print(
        f"diagnosed {n_new} new rows in {time.perf_counter() - t0:.0f}s "
        f"({len(records) + max(0, len(rows) - n_new)} total in file)",
        file=sys.stderr,
    )
    return 0


# --- the Markdown report -------------------------------------------------------------
def load_all_records(out_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every diagnosed row (the file may hold rows from earlier resumed runs)."""
    by_id: dict[str, dict[str, Any]] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                by_id[str(rec["id"])] = rec
            except ValueError:
                continue
    return [by_id[str(r["id"])] for r in rows if str(r["id"]) in by_id]


def _md_table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return lines


def render_report(
    records: list[dict[str, Any]],
    scanned: dict[str, Any],
    args: argparse.Namespace,
    zero_docs: list[str] | None = None,
) -> str:
    no_answer = [r for r in records if r["status"] != "accepted"]
    accepted = [r for r in records if r["status"] == "accepted"]
    classes = ["R0", "R1", "R2", "R3", "R4"]
    L: list[str] = []
    L.append("# MMLongBench-Doc no-answer diagnosis - dev (ticket 274)")
    L.append("")
    L.append(
        f"rows: {len(records)} in-scope answerable dev rows (tag `{args.tag}`). "
        f"{len(no_answer)} got no answer (none/check_failed) and were classified "
        f"R0-R4 below; {len(accepted)} were accepted by the lane."
    )
    L.append("")
    L.append(
        "R0 = question not self-contained (Qwen judge, question text alone); "
        "R1 = the gold evidence page is not in the lane's passages (no page by "
        "metadata AND the gold answer not in any gold-doc passage); "
        "R2 = the evidence page is in the passages but the gold answer is not in "
        "the extracted text; R3 = the gold answer IS in the gold-doc passage text "
        "but the lane declined; R4 = other / cannot tell. The lane was re-asked "
        "for the no-answer rows (~1 s each), so `rejected_failed` and "
        "`status_reask` come from the fresh call."
    )
    L.append("")
    L.append("## Class counts (the 62 no-answer rows)")
    L.append("")
    body = [[c, str(sum(1 for r in no_answer if r["class"] == c))] for c in classes]
    L += _md_table(["class", "n"], body)
    L.append("")
    for axis, key in (("evidence source", "stratum"), ("answer_format", "answer_format")):
        keys = sorted({str(r[key]) for r in no_answer})
        L.append(f"## By {axis}")
        L.append("")
        body = [
            [k, str(sum(1 for r in no_answer if str(r[key]) == k))]
            + [
                str(sum(1 for r in no_answer if str(r[key]) == k and r["class"] == c))
                for c in classes
            ]
            for k in keys
        ]
        L += _md_table([axis, "n"] + classes, body)
        L.append("")
    L.append(
        "the `chart_image` rows above carry EMPTY `evidence_sources` in the parquet "
        "(the stratum fallback label); they are answerable rows, not vision-only ones"
    )
    L.append("")
    L.append("## R0 over ALL in-scope rows (the shared-corpus ceiling)")
    L.append("")
    r0_all = [r for r in records if r["self_contained"] is False]
    r0_na = [r for r in no_answer if r["self_contained"] is False]
    r0_acc = [r for r in accepted if r["self_contained"] is False]
    L.append(
        f"{len(r0_all)}/{len(records)} in-scope dev questions are NOT self-contained "
        f"(judge, question text alone): {len(r0_na)} among the no-answer rows, "
        f"{len(r0_acc)} among the accepted rows. No shared-corpus run can do better "
        "than this bound on those questions - the official task gives one PDF per "
        "question, but the 25-doc corpus does not."
    )
    L.append("")
    L.append("## Accepted rows (the 20)")
    L.append("")
    correct = sum(1 for r in accepted if r.get("correct") is True)
    partial = sum(1 for r in accepted if r.get("partial") is True)
    wrong = sum(1 for r in accepted if r.get("correct") is False and not r.get("partial"))
    L.append(
        f"{len(accepted)} accepted: {correct} fully correct (official score 1.0), "
        f"{partial} partial credit, {wrong} scored 0."
    )
    L.append("")
    r1 = [r for r in no_answer if r["class"] == "R1"]
    if r1:
        L.append("## R1 and scanned pages")
        L.append("")
        rows_with_scanned = 0
        ev_pages_total = ev_pages_scanned = 0
        for r in r1:
            sc = (scanned.get(r["doc_id"]) or {}).get("scanned") or []
            ev = [p for p in r["evidence_pages"] if p > 0]
            if any(p in sc for p in ev):
                rows_with_scanned += 1
            ev_pages_total += len(ev)
            ev_pages_scanned += sum(1 for p in ev if p in sc)
        share = rows_with_scanned / len(r1)
        page_share = ev_pages_scanned / ev_pages_total if ev_pages_total else 0.0
        L.append(
            f"{rows_with_scanned}/{len(r1)} R1 rows have at least one gold evidence "
            f"page that is a scanned page (<{bm.SCAN_MIN_CHARS} chars/page, pypdf, "
            f"recomputed per page): a {share:.0%} share of R1. Over the R1 rows' "
            f"{ev_pages_total} evidence pages, {ev_pages_scanned} are scanned "
            f"({page_share:.0%}). For the rest, the gold page has text in the DB "
            "but retrieval did not put it in the lane's passages."
        )
        L.append("")
    cf = [r for r in no_answer if r["status"] == "check_failed"]
    if cf:
        L.append("## check_failed: which check failed (re-asked DocResult.rejected)")
        L.append("")
        counter = Counter(r["rejected_failed"] or "(none recorded)" for r in cf)
        for reason, n in counter.most_common(3):
            L.append(f"- {reason}: {n}/{len(cf)}")
        L.append("")
    L.append("## Examples (up to 5 per class)")
    L.append("")
    for c in classes:
        ex = [r for r in no_answer if r["class"] == c][:N_EXAMPLES]
        if not ex:
            continue
        L.append(f"### {c} ({sum(1 for r in no_answer if r['class'] == c)} rows)")
        L.append("")
        if c == "R2":
            n_derived = sum(
                1 for r in no_answer if r["class"] == "R2" and r.get("answer_in_doc_text") == "N"
            )
            n_r2 = sum(1 for r in no_answer if r["class"] == "R2")
            L.append(
                f"of the {n_r2} R2 rows, {n_derived} have a gold answer that never appears "
                "anywhere in the full gold-doc DB text - derived/calculated values "
                "(ratios, sums of line items, differences) the lane would have had to "
                "compute from the input figures on the evidence pages, not a "
                "text-extraction failure"
            )
            L.append("")
        for r in ex:
            retrieved = r["retrieved_pages_of_gold_doc"] or "(none)"
            if r["whole_file"]:
                retrieved = "whole file"
            L.append(
                f"- **{r['id']}** (gold `{r['doc_id'][:12]}...`, evidence pages "
                f"{r['evidence_pages']}, {r['answer_format']}) — "
                f"Q: {r['question'].strip()}"
            )
            L.append(f"  - gold: {r['gold']}")
            L.append(
                f"  - retrieved: files {', '.join(r['top_files_retrieved']) or '(none)'}; "
                f"gold-doc pages {retrieved}; gold answer in gold-doc passages: "
                f"{r['gold_answer_in_gold_passages']} ({r['answer_how'] or 'not found'})"
            )
            if r.get("answer_in_doc_text") == "N":
                L.append(
                    "  - gold answer in the FULL gold-doc DB text: N (the value never "
                    "appears anywhere in the document - a derived/calculated value, or "
                    "a property the document does not state in text)"
                )
            if r["rejected_failed"]:
                L.append(f"  - failed check: {r['rejected_failed']}")
            if r["status_reask"] and r["status_reask"] != r["status"]:
                L.append(f"  - re-ask: {r['status_reask']}")
            L.append(f"  - judge: self_contained={r['self_contained']} ({r['self_contained_why']})")
            L.append("")
    L.append("")
    if zero_docs:
        L.append("## Corpus documents with zero extractable text")
        L.append("")
        affected = [r for r in records if r["doc_id"] in set(zero_docs)]
        L.append(
            f"{len(zero_docs)} of the 25 corpus documents have ZERO chunks in the DB "
            f"(the reader extracted no text at all - image-only slide decks): "
            + ", ".join(f"`{d}`" for d in zero_docs)
            + f". Every in-scope dev row about them ({len(affected)}: "
            + ", ".join(r["id"] for r in affected)
            + ") is unanswerable by ANY text lane, retrieval or not; their R0/R1 "
            "class rests on the question text, but the empty document is a second, "
            "independent cause."
        )
        L.append("")
    L.append("## What could not be decided")
    L.append("")
    L.append(
        "R0 rests on a Qwen yes/no judge (printed above per example, hand-readable) "
        "- borderline questions are the judge's call, and the judge sees the question "
        "text only, as any shared-corpus retriever would. R2 vs R3 for rows whose "
        "gold evidence page is a scanned page cannot be separated further without OCR "
        "(ticket T5): the page is in the context but its text is empty/fragments. "
        "The check_failed reasons come from the RE-ASKED lane call, not the original "
        "run (the original run file does not record `rejected`); a re-ask can in "
        "principle land on a different check, and any such flip is noted per row in "
        "the jsonl (`status_reask`)."
    )
    return "\n".join(L) + "\n"


# --- CLI ---------------------------------------------------------------------------------
def parse_args(argv: Any = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="bench_mmlongdoc_diagnose",
        description="MMLongBench-Doc no-answer diagnosis (ticket 274, read-only)",
    )
    ap.add_argument("--tag", default="dev", help="the run/score tag (default dev)")
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="diagnose jsonl (default out/mmlongdoc/diagnose-dev.jsonl)",
    )
    ap.add_argument(
        "--md",
        default=str(DEFAULT_MD),
        help="Markdown report (default out/mmlongdoc/diagnose-dev.md)",
    )
    ap.add_argument("--db", default=str(bm.DEFAULT_DB), help="corpus DB path")
    ap.add_argument("--corpus-dir", default=str(bm.DEFAULT_CORPUS_DIR))
    ap.add_argument(
        "--data-dir",
        default=str(bm.DEFAULT_DATA_DIR),
        help="raw data root for the per-page scanned scan (~/bench_data/mmlongdoc)",
    )
    ap.add_argument("--today", default=bm.DEFAULT_TODAY)
    ap.add_argument("--budget-s", type=float, default=bm.DOC_BUDGET_S)
    ap.add_argument("--base-url", default=bm.base_url_default(), help="the Qwen relay base URL")
    ap.add_argument("--model", default=bm.DEFAULT_MODEL)
    ap.add_argument(
        "--skip-reask",
        action="store_true",
        help="do not re-ask the lane (no fresh `rejected`; slower classes rest on "
        "the original status only)",
    )
    return ap.parse_args(argv)


def main(argv: Any = None) -> int:
    args = parse_args(argv)
    src_dir = str(REPO_ROOT / "src")
    pipe_dir = str(REPO_ROOT / "pipeline" / "src")
    for d in (src_dir, pipe_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    rows = in_scope_dev_rows()
    if not rows:
        print("no in-scope dev rows in the questions file", file=sys.stderr)
        return 2
    if not args.skip_reask:
        try:
            asyncio.run(diagnose_rows(args))
        except bm.RunAbortedError as exc:
            print(f"aborted: {exc}", file=sys.stderr)
            return 1
    # the report always rebuilds from the jsonl (resumable, deterministic)
    out_path = Path(args.out)
    if not out_path.is_file():
        print(f"no diagnose records at {out_path}; run without --skip-reask first", file=sys.stderr)
        return 2
    manifest = json.loads(bm.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    scanned = scanned_pages_map(Path(args.data_dir).expanduser(), manifest)
    records = load_all_records(out_path, rows)
    # backfill the whole-doc DB check for records written before it existed
    # (deterministic, no relay call)
    db = Path(args.db)
    cache: dict[tuple[str, str, str], bool | None] = {}
    for r in records:
        if "answer_in_doc_text" in r:
            continue
        key = (str(r["doc_id"]), str(r["gold"]), str(r["answer_format"]))
        if key not in cache:
            cache[key] = answer_in_gold_doc_db(db, r["doc_id"], r["gold"], r["answer_format"])
        r["answer_in_doc_text"] = "Y" if cache[key] else ("N" if cache[key] is not None else None)
    # persist the enrichment back into the jsonl (deterministic; the file stays
    # self-describing after a --skip-reask rebuild). Any row in the file that is
    # not one of the current in-scope rows is kept, appended after.
    existing: list[dict[str, Any]] = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            existing.append(json.loads(line))
        except ValueError:
            continue  # a torn line is dropped; the row is simply re-run
    known = {str(r["id"]) for r in records}
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records + [e for e in existing if str(e.get("id")) not in known]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = render_report(records, scanned, args, zero_docs=zero_chunk_docs(db))
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
