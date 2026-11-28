"""MMLongBench-Doc offline benchmark adapter and scorer bridge (tickets 272/273).

Producer for the ticket-269 A/B: one command samples the pinned MMLongBench-Doc
revision into a seeded manifest. Since ticket 273 the sample is DOCUMENT-FIRST:
the official task is one PDF per question, so 25 documents are chosen first
(seeded sha256 rule, >=6-usable-question documents preferred, ~1,200-page cap,
text/table/layout/unanswerable-bearing coverage) and then ~200 questions are
sampled from those documents only (the 272 strata/quota/split rules unchanged).
The old 110-PDF 272 sample is kept as ``manifest-110.json`` /
``questions-110.jsonl``. The manifest's PDFs (each content-hash verified
against the pinned revision) are ingested into ONE corpus DB
(``out/mmlongdoc/corpus25/``, flat folder, no --ocr). It answers each
question through the REAL document lane (``voice_pipeline.ask_documents.ask_documents``
in docs-only mode: no web, no
supplement, no fallback) with the local Qwen relay as the answerer, and scores the
spoken answers with the vendored OFFICIAL rules
(``scripts/mmlongdoc_eval_score.py``, Apache-2.0, logic unchanged) after a Qwen
typed-answer extraction (the official GPT-4o extractor prompt, adapted to JSON out).

Subcommands (run from the repo root; full guide in docs/MMLONGDOC-BENCH.md):

  sample             write eval/mmlongdoc/manifest.json + questions.jsonl
                     (the 25-doc corpus; document-first, then question sampling)
  fetch              download the manifest's PDFs, verify their content hash, record
                     scanned-page share (<20 chars/page via pypdf) into the manifest
  ingest             one corpus DB for the sampled PDFs (records seconds, file count)
  run --split S      answer the split's questions through ask_documents; per-question
                     jsonl at out/mmlongdoc/run-<tag>.jsonl, resumable (skips done ids)
  score              Qwen extraction + official rules; out/mmlongdoc/score-<tag>.json
                     + a Markdown table
  calibration-sheet  50 stratified seeded rows (question, gold, spoken, Qwen's
                     extracted answer, the rules' verdict) for the orchestrator's
                     hand audit - this script does not grade them

Free: the local Qwen relay only (no Gemini, no web search, no paid call of any
kind). ``SUFFICIENCY`` stays False; nothing here modifies ask_documents,
agent_server or sufficiency. Raw data lives in ~/bench_data/mmlongdoc/ and run
artifacts in out/mmlongdoc/ - neither is ever committed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- pinned dataset ---------------------------------------------------------
HF_REPO = "yubo2333/MMLongBench-Doc"
HF_REVISION = "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
PARQUET_PATH = "data/train-00000-of-00001.parquet"
DOCS_PREFIX = "documents/"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}"

DEFAULT_DATA_DIR = Path("~/bench_data/mmlongdoc").expanduser()
EVAL_DIR = REPO_ROOT / "eval" / "mmlongdoc"
OUT_DIR = REPO_ROOT / "out" / "mmlongdoc"
DEFAULT_MANIFEST = EVAL_DIR / "manifest.json"
DEFAULT_QUESTIONS = EVAL_DIR / "questions.jsonl"
# ticket 273: the corpus is the 25-doc one (the 272 110-doc DB stays at
# out/mmlongdoc/corpus.db, reachable with explicit --db/--corpus-dir)
DEFAULT_DB = OUT_DIR / "corpus25" / "corpus.db"
DEFAULT_CORPUS_DIR = OUT_DIR / "corpus25" / "corpus"

# --- seeded sha256 rules (the implicit-split:7 convention, own salts) --------
SEED = "7"
TOTAL = 200
# ticket 273: document-first corpus pick
DOC_N = 25
DOC_PAGE_BUDGET = 1200
DOC_PRIORITY_MIN_USABLE = 6  # prefer documents with >= this many usable questions
UNANSWERABLE_FRACTION = 0.2  # the unanswerable quota scales to ~20% of the sample
CALIBRATION_N = 50

SCAN_MIN_CHARS = 20  # a page with <20 extracted chars counts as scanned
DEFAULT_TODAY = "2026-11-28"  # the fixed --today the lane's clock sees
DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DOC_BUDGET_S = 60.0
CONSECUTIVE_ERROR_LIMIT = 3
USER_AGENT = "me4-bench-272"

NOT_ANSWERABLE = "Not answerable"
TEXT_SOURCE = "Pure-text (Plain-text)"
TABLE_SOURCE = "Table"
LAYOUT_SOURCE = "Generalized-text (Layout)"
VISION_SOURCES = frozenset({"Chart", "Figure"})
STRATA_ORDER = ("text", "table", "layout", "chart_image", "unanswerable")

# The official GPT-4o extractor prompt, VERBATIM from
# https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/eval/prompt_for_answer_extraction.md
# (fetched 2026-11-28). The only adaptation is the JSON output instruction appended
# at the call site (the official flow pre-fills the assistant turn; the relay call
# here is one user message).
OFFICIAL_EXTRACTOR_PROMPT = "Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. \n- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type \"Not answerable\". Exception: If the analysis only tells you that it can not read/understand the images or documents, type \"Fail to answer\".\n- Please make your response as concise as possible. Also note that your response should be formatted as below:\n```\nExtracted answer: [answer]\nAnswer format: [answer format]\n```\n\nPlease read the following example, then extract the answer from the model response and type it at the end of the prompt. \n\n---\nQuestion: List the primary questions asked about the services in this report.\nAnalysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\\n\\n1. Is the service safe?\\n2. Is the service effective?\\n3. Is the service caring?\\n4. Is the service responsive?\\n5. Is the service well-led?\nExtracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']\nAnswer format: List\n\n---\nQuestion: How many regulations of the HSCA 2008 are breached in all according to this report?\nAnalysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\\n\\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\\n2. Regulation 12: Safe care and treatment\\n3. Regulation 18: Staffing\\n4. Regulation 11: Need for consent\\n5. Regulation 10: Dignity and respect\\n6. Regulation 9: Person-centred care\\n7. Regulation 17: Good governance\\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\\n\\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.\nExtracted answer: 10\nAnswer format: Integer\n\n---\nQuestion: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?\nAnalysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!\nExtracted answer: Not answerable\nAnswer format: String\n\n---\nQuestion: How many quotations from male respondent over 50 years old are included in this report?\nAnalysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.\nExtracted answer: Fail to answer\nAnswer format: String\n\n---"  # noqa: E501 (the official prompt, one verbatim string)

EXTRACTOR_SYSTEM = "You are an exact, concise answer extractor for document-QA evaluation."
EXTRACT_JSON_INSTRUCTION = (
    "\nExtract the answer now. Reply with JSON only: an object with the keys "
    '"extracted_answer" (the concise answer - a list for List format, or the string '
    f'"{NOT_ANSWERABLE}" when the documents do not state it) and "answer_format" '
    '(one of "Integer", "Float", "String", "List", "Not answerable", "Fail to answer").'
)


class RunAbortedError(RuntimeError):
    """The relay is failing en masse; stop loudly instead of scoring errors."""


# --- seeded rules (pure) -----------------------------------------------------
def sha_rank(prefix: str, qid: str) -> int:
    """The seeded sha256 rank: ``int(sha256(prefix:qid), 16)`` (the implicit-split:7
    convention, with this bench's own salts)."""
    return int(hashlib.sha256(f"{prefix}:{qid}".encode()).hexdigest(), 16)


def sample_rank(qid: str, seed: str = SEED) -> int:
    """Within-stratum selection rank: lower rank is sampled first."""
    return sha_rank(f"mmlongdoc-sample:{seed}", qid)


def split_for(qid: str, seed: str = SEED) -> str:
    """The dev/test split: dev iff the rank is even parity (the implicit-split:7 rule)."""
    return "dev" if sha_rank(f"mmlongdoc-split:{seed}", qid) % 2 == 0 else "test"


def doc_rank(doc_id: str, seed: str = SEED) -> int:
    """Document selection rank (ticket 273): lower rank is chosen first
    (the seeded sha256 convention, own salt)."""
    return sha_rank(f"mmlongdoc-doc:{seed}", doc_id)


def is_unanswerable(answer: Any) -> bool:
    return str(answer or "").strip() == NOT_ANSWERABLE


def parse_list_field(value: Any) -> list[str]:
    """The parquet stores evidence_pages/evidence_sources as Python-repr strings
    (``"[5]"`` / ``"['Chart']"``); real lists pass through. Never raises."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    s = str(value or "").strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def parse_pages(value: Any) -> list[int]:
    out: list[int] = []
    for v in parse_list_field(value):
        try:
            out.append(int(v))
        except ValueError:
            out.append(-1)  # unparseable page: recorded, not silently dropped
    return out


def is_vision_only(sources: Sequence[str]) -> bool:
    s = set(sources)
    return bool(s) and s <= VISION_SOURCES


def evidence_stratum(sources: Sequence[str]) -> str:
    """The evidence stratum of an answerable row: ``text``/``table``/``layout`` by
    priority, or ``chart_image`` when the evidence is vision-only (Chart/Figure) -
    or unrecognised, which is conservatively out of scope."""
    s = set(sources)
    if not s or s <= VISION_SOURCES:
        return "chart_image"
    if TEXT_SOURCE in s:
        return "text"
    if TABLE_SOURCE in s:
        return "table"
    if LAYOUT_SOURCE in s:
        return "layout"
    return "chart_image"


def annotate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach the deterministic fields: qid (the row index in the pinned parquet),
    the parsed evidence fields, the unanswerable flag, the strata and the vision flag."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        sources = parse_list_field(r.get("evidence_sources"))
        unans = is_unanswerable(r.get("answer"))
        estratum = "unanswerable" if unans else evidence_stratum(sources)
        doc_type = str(r.get("doc_type") or "")
        out.append(
            {
                "qid": f"q{i:04d}",
                "doc_id": str(r.get("doc_id") or ""),
                "doc_type": doc_type,
                "question": str(r.get("question") or ""),
                "answer": str(r.get("answer") or ""),
                "answer_format": str(r.get("answer_format") or ""),
                "evidence_pages": parse_pages(r.get("evidence_pages")),
                "evidence_sources": sources,
                "unanswerable": unans,
                "evidence_stratum": estratum,
                "out_of_scope_vision": (not unans) and is_vision_only(sources),
                # the sampling stratum: unanswerable is its own stratum; the rest are
                # stratified by (evidence, doc_type)
                "stratum": "unanswerable" if unans else f"{estratum}:{doc_type}",
            }
        )
    return out


def doc_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-document corpus stats from annotated rows (ticket 273): the question
    counts (n / usable / unanswerable), the evidence capabilities the doc
    carries (``T``=text, ``B``=table, ``L``=layout evidence in any of its
    questions, ``U``=it bears an unanswerable question), and ``max_page`` —
    the largest evidence page, the deterministic page estimate used when the
    pinned PDF is not local."""
    stats: dict[str, dict[str, Any]] = {}
    for r in rows:
        doc_id = str(r.get("doc_id") or "")
        if not doc_id:
            continue
        s = stats.setdefault(
            doc_id, {"n": 0, "usable": 0, "unans": 0, "caps": set(), "max_page": 0}
        )
        s["n"] += 1
        if r["unanswerable"]:
            s["unans"] += 1
            s["caps"].add("U")
        elif not r["out_of_scope_vision"]:
            s["usable"] += 1
        sources = set(r["evidence_sources"])
        if TEXT_SOURCE in sources:
            s["caps"].add("T")
        if TABLE_SOURCE in sources:
            s["caps"].add("B")
        if LAYOUT_SOURCE in sources:
            s["caps"].add("L")
        for p in r["evidence_pages"]:
            if p > 0:
                s["max_page"] = max(s["max_page"], p)
    return stats


def allocate_quotas(population: Mapping[str, int], seats: int, minimum: int = 1) -> dict[str, int]:
    """Largest-remainder proportional allocation of ``seats`` over the non-empty
    strata with a per-stratum floor of ``minimum`` (dropped when the seats cannot
    afford every floor), never exceeding a stratum's population. Deterministic:
    remainder points go to the largest fractional part, ties break on the stratum
    key. Requires ``seats <= sum(population)``."""
    keys = sorted(k for k, n in population.items() if n > 0)
    if not keys or seats <= 0:
        return {k: 0 for k in population}
    total = sum(population[k] for k in keys)
    if seats > total:
        raise RuntimeError(f"seats {seats} exceed the available rows {total}")
    floors = {k: min(minimum, population[k]) for k in keys}
    if sum(floors.values()) > seats:
        floors = {k: 0 for k in keys}
    exact = {k: (seats - sum(floors.values())) * population[k] / total for k in keys}
    quota = {k: floors[k] + int(exact[k]) for k in keys}
    leftover = int(round(seats - sum(floors.values()) - sum(int(exact[k]) for k in keys)))
    ranked = sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k))
    for k in ranked:
        if leftover <= 0:
            break
        if quota[k] < population[k]:
            quota[k] += 1
            leftover -= 1
    return {k: quota.get(k, 0) for k in population}


def select_documents(
    rows: Sequence[Mapping[str, Any]],
    seed: str = SEED,
    n_docs: int = DOC_N,
    page_budget: int = DOC_PAGE_BUDGET,
    pages: Mapping[str, int] | None = None,
) -> list[str]:
    """The document-first seeded corpus pick (ticket 273): pick DOCUMENTS, then
    questions are sampled from them only (see ``select_sample``). Deterministic
    in (rows, seed, n_docs, page_budget, pages); returns the chosen doc_ids,
    sorted.

    1. Candidates are ranked (prefer documents with >= ``DOC_PRIORITY_MIN_USABLE``
       usable questions, then the seeded ``doc_rank``, then doc_id) and the top
       ``n_docs`` are taken.
    2. Page cap: while the chosen documents' pages exceed ``page_budget``, the
       largest chosen doc is swapped for the smallest not-chosen doc (the
       >=6-usable tier first) — only when the swap lowers the total AND keeps
       every evidence capability (text/table/layout/unanswerable-bearing) the
       dataset as a whole carries; otherwise the loop stops and the real total
       is reported.
    3. Coverage safety net: any capability still missing from the pick gets
       the highest-ranked doc carrying it (evicting the largest chosen doc
       whose removal keeps coverage; adding over the cap when nothing is
       evictable).

    ``pages`` maps doc_id -> real page count (the pinned PDF's); missing entries
    fall back to the deterministic parquet estimate (max evidence page).
    """
    stats = doc_stats(rows)
    pages = pages or {}

    def pof(doc_id: str) -> int:
        p = pages.get(doc_id)
        if p is not None and int(p) > 0:
            return int(p)
        return int(stats[doc_id]["max_page"] or 1)

    def is_priority(doc_id: str) -> bool:
        return stats[doc_id]["usable"] >= DOC_PRIORITY_MIN_USABLE

    order = {d: (0 if is_priority(d) else 1, doc_rank(d, seed), d) for d in stats}
    cands = sorted(stats, key=lambda d: order[d])
    selected = cands[: min(n_docs, len(cands))]

    def coverage(doc_ids: Sequence[str]) -> set[str]:
        out: set[str] = set()
        for d in doc_ids:
            out |= stats[d]["caps"]
        return out

    required = set().union(*(s["caps"] for s in stats.values())) if stats else set()

    # the page cap (step 2)
    while sum(pof(d) for d in selected) > page_budget:
        chosen = set(selected)
        not_chosen = [d for d in cands if d not in chosen]
        if not not_chosen:
            break
        victim = max(selected, key=lambda d: (pof(d), order[d]))
        tier = [d for d in not_chosen if is_priority(d)]
        pool = tier if tier else not_chosen
        swapped = False
        for repl in sorted(pool, key=lambda d: (pof(d), order[d])):
            if pof(repl) >= pof(victim):
                break  # the smallest not-chosen doc already does not help
            after = (coverage(selected) - stats[victim]["caps"]) | stats[repl]["caps"]
            if not required <= after:
                continue  # would drop a capability the dataset carries
            selected = [repl if d == victim else d for d in selected]
            swapped = True
            break
        if not swapped:
            break

    # the coverage safety net (step 3; a no-op when the top pick covers all)
    for cap in ("T", "B", "L", "U"):
        if cap in coverage(selected):
            continue
        add = [d for d in cands if d not in set(selected) and cap in stats[d]["caps"]]
        if not add:
            continue
        d = add[0]
        victim = None
        for s in sorted(selected, key=lambda s: (pof(s), order[s]), reverse=True):
            after = (coverage(selected) - stats[s]["caps"]) | stats[d]["caps"]
            if required <= after:
                victim = s
                break
        if victim is not None:
            selected = [d if x == victim else x for x in selected]
        else:
            selected.append(d)  # over the cap; coverage wins, the total is reported
    return sorted(selected)


def select_sample(
    rows: Sequence[Mapping[str, Any]],
    seed: str = SEED,
    total: int = TOTAL,
    unanswerable_quota: int = 40,
) -> list[dict[str, Any]]:
    """The seeded stratified sample (deterministic in rows/seed/total/quota):
    ``unanswerable`` gets its fixed quota, the rest is allocated proportionally
    over the ``(evidence, doc_type)`` strata, and within each stratum the rows are
    ranked by ``sample_rank(qid)`` - the top ``quota`` rows are taken. Every
    selected row gets its dev/test ``split`` from ``split_for``."""
    population: Counter[str] = Counter(str(r["stratum"]) for r in rows)
    unans_take = min(unanswerable_quota, population.get("unanswerable", 0))
    other = {k: n for k, n in population.items() if k != "unanswerable"}
    quotas = allocate_quotas(other, total - unans_take)
    quotas["unanswerable"] = unans_take
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["stratum"]), []).append(r)
    selected: list[dict[str, Any]] = []
    for stratum, group in groups.items():
        take = quotas.get(stratum, 0)
        if take <= 0:
            continue
        ranked = sorted(group, key=lambda r: (sample_rank(str(r["qid"]), seed), str(r["qid"])))
        for r in ranked[:take]:
            selected.append(dict(r, split=split_for(str(r["qid"]), seed)))
    selected.sort(key=lambda r: str(r["qid"]))
    return selected


def count_sample(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(r["split"]) for r in selected)
    by_stratum = Counter(str(r["stratum"]) for r in selected)
    by_evidence = Counter(str(r["evidence_stratum"]) for r in selected)
    by_doc_type = Counter(str(r["doc_type"]) for r in selected)
    by_format = Counter(str(r["answer_format"]) for r in selected)
    return {
        "total": len(selected),
        "by_split": {k: by_split.get(k, 0) for k in ("dev", "test")},
        "by_stratum": dict(sorted(by_stratum.items())),
        "by_evidence": {k: by_evidence.get(k, 0) for k in STRATA_ORDER},
        "by_doc_type": dict(sorted(by_doc_type.items())),
        "by_answer_format": dict(sorted(by_format.items())),
        "in_scope": sum(
            1 for r in selected if not r["out_of_scope_vision"] and not r["unanswerable"]
        ),
        "out_of_scope_vision": sum(1 for r in selected if r["out_of_scope_vision"]),
        "unanswerable": sum(1 for r in selected if r["unanswerable"]),
    }


# --- parquet / HF / PDF plumbing ---------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_file(path: Path, algo: str) -> str:
    """The manifest's per-PDF hash: content sha256 (LFS files) or the git blob id
    (non-LFS: sha1 of ``"blob <size>\\0<content>"``, which pins size and content)."""
    if algo == "sha256":
        return sha256_file(path)
    if algo == "git_sha1":
        data = path.read_bytes()
        h = hashlib.sha1()
        h.update(b"blob %d\x00" % len(data))
        h.update(data)
        return h.hexdigest()
    raise ValueError(f"unknown hash algo: {algo}")


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """The pinned parquet as plain rows (pyarrow only; the rest of the bench and its
    tests do not need it, so it is imported lazily)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is needed to read the parquet: run under `uv run --with pyarrow` "
            "(see docs/MMLONGDOC-BENCH.md)"
        ) from exc
    table = pq.read_table(str(path))
    return [dict(row) for row in table.to_pylist()]


def hf_url(rel_path: str) -> str:
    return f"{HF_BASE_URL}/{rel_path}"


def hf_document_listing() -> dict[str, dict[str, str]]:
    """path -> {"hash", "algo"} for every documents/*.pdf at the pinned revision, from
    the HF tree API (free, no key). LFS files: the lfs oid is the content sha256.
    Non-LFS files: the tree oid is the git blob sha1 (sha1 of "blob <size>\\0<content>"),
    which also pins the exact content (and its size)."""
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}/documents"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    out: dict[str, dict[str, str]] = {}
    for item in items:
        path = str(item.get("path") or "")
        if not path.startswith(DOCS_PREFIX) or not path.endswith(".pdf"):
            continue
        lfs = item.get("lfs") or {}
        if lfs.get("oid"):
            out[path] = {"hash": str(lfs["oid"]), "algo": "sha256"}
        elif item.get("oid"):
            out[path] = {"hash": str(item["oid"]), "algo": "git_sha1"}
    return out


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"download failed after 3 attempts: {url} ({last})")


def pdf_page_count(path: Path) -> int:
    """The page count of a local PDF (pypdf, no text extraction — the doc
    selection's page cap only needs the count, not the text)."""
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def scan_pdf(path: Path) -> tuple[int, int]:
    """(total_pages, scanned_pages): a page is scanned when pypdf extracts fewer than
    ``SCAN_MIN_CHARS`` characters of text from it (the <20 chars/page heuristic)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    scanned = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if len(text.strip()) < SCAN_MIN_CHARS:
            scanned += 1
    return total, scanned


def scanned_block(
    pdf_paths: Sequence[Path], on_pdf: Callable[[Path, int, int], None] | None = None
) -> dict[str, Any]:
    total = scanned = 0
    per_pdf: list[dict[str, Any]] = []
    for p in pdf_paths:
        t, s = scan_pdf(p)
        total += t
        scanned += s
        per_pdf.append({"path": str(p.name), "pages": t, "scanned_pages": s})
        if on_pdf is not None:
            on_pdf(p, t, s)
    return {
        "min_chars_per_page": SCAN_MIN_CHARS,
        "total_pages": total,
        "scanned_pages": scanned,
        "share": round(scanned / total, 4) if total else None,
        "per_pdf": per_pdf,
    }


def _load_module(name: str, filename: str) -> Any:
    """Import a sibling scripts/ file (the sufficiency_eval helpers, the judge's
    key reader, the vendored official rules) without package machinery."""
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sufficiency_eval() -> Any:
    """scripts/sufficiency_eval.py: the Relay, relay_payload, JSON_ONLY_SUFFIX, the
    manila_now clock helper and the DB card/file loaders (the judge_eval convention)."""
    return _load_module("sufficiency_eval", "sufficiency_eval.py")


def load_judge_eval() -> Any:
    """scripts/judge_eval.py: read_key (JUDGE_API_KEY or ~/.secrets/api_key.txt)."""
    return _load_module("judge_eval", "judge_eval.py")


def load_official() -> Any:
    """scripts/mmlongdoc_eval_score.py: the vendored official rules (unchanged)."""
    return _load_module("mmlongdoc_eval_score", "mmlongdoc_eval_score.py")


def read_key() -> str:
    return str(load_judge_eval().read_key())


# --- sample -------------------------------------------------------------------
def cmd_sample(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser()
    eval_dir = Path(args.eval_dir)
    parquet = data_dir / PARQUET_PATH
    if not parquet.is_file():
        print(
            f"parquet not found at {parquet}; downloading {HF_REPO}@{HF_REVISION} (free)",
            file=sys.stderr,
        )
        download_file(hf_url(PARQUET_PATH), parquet)
    rows = annotate(load_parquet_rows(parquet))
    parquet_sha = sha256_file(parquet)
    n_rows = len(rows)
    stats = doc_stats(rows)
    try:
        listing = hf_document_listing()
    except Exception as exc:  # the API is free and keyless; offline -> local shas
        listing = {}
        print(
            f"warning: HF tree API unreachable ({exc}); using local file sha256 for the PDF list",
            file=sys.stderr,
        )

    # --- 1. pick the documents (ticket 273: document-first) --------------------
    # page counts for the page cap: the real count of the pinned PDF where
    # possible (local, or fetched free for the >=6-usable tier), else the
    # deterministic parquet estimate (max evidence page); per-doc source below
    pages: dict[str, int] = {}
    page_source: dict[str, str] = {}
    for doc_id in sorted(stats):
        s = stats[doc_id]
        local = data_dir / DOCS_PREFIX / doc_id
        if local.is_file():
            try:
                pages[doc_id] = pdf_page_count(local)
                page_source[doc_id] = "local"
                continue
            except Exception as exc:
                print(
                    f"warning: cannot count pages of {doc_id} ({exc}); estimating",
                    file=sys.stderr,
                )
        elif s["usable"] >= DOC_PRIORITY_MIN_USABLE and listing:
            path = f"{DOCS_PREFIX}{doc_id}"
            entry = listing.get(path)
            if entry is not None:
                try:
                    download_file(hf_url(path), local)
                    if hash_file(local, entry["algo"]) == entry["hash"]:
                        pages[doc_id] = pdf_page_count(local)
                        page_source[doc_id] = "downloaded"
                        continue
                    local.unlink(missing_ok=True)
                    print(
                        f"warning: {entry['algo']} mismatch for {path}; estimating",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    print(
                        f"warning: cannot fetch {path} for the page count ({exc}); estimating",
                        file=sys.stderr,
                    )
        pages[doc_id] = int(s["max_page"] or 1)
        page_source[doc_id] = "estimated"
    chosen_docs = select_documents(
        rows, seed=args.seed, n_docs=args.docs, page_budget=args.page_budget, pages=pages
    )
    docs_pages_total = sum(pages[d] for d in chosen_docs)
    pool = [r for r in rows if r["doc_id"] in set(chosen_docs)]
    if len(pool) < args.total:
        print(
            f"note: the pool of the {len(chosen_docs)} chosen documents has only {len(pool)} "
            f"questions (< {args.total}); using all of them, no padding",
            file=sys.stderr,
        )
    total_target = min(args.total, len(pool))
    pool_unans = sum(1 for r in pool if r["unanswerable"])
    if args.unanswerable is None:
        unans_quota = min(pool_unans, int(round(total_target * UNANSWERABLE_FRACTION)))
    else:
        unans_quota = min(args.unanswerable, pool_unans)

    # --- 2. sample the questions from those documents (the 272 rules) ----------
    selected = select_sample(
        pool, seed=args.seed, total=total_target, unanswerable_quota=unans_quota
    )
    pdfs: list[dict[str, str]] = []
    for doc_id in chosen_docs:
        path = f"{DOCS_PREFIX}{doc_id}"
        entry = listing.get(path)
        local = data_dir / path
        if entry is None:
            if local.is_file():
                entry = {"hash": sha256_file(local), "algo": "sha256"}
            else:
                raise RuntimeError(
                    f"no hash for {path}: the HF API is unreachable and the file is not local"
                )
        pdfs.append({"path": path, "doc_id": doc_id, "hash": entry["hash"], "algo": entry["algo"]})
    local_ok = all(
        (data_dir / p["path"]).is_file() and hash_file(data_dir / p["path"], p["algo"]) == p["hash"]
        for p in pdfs
    )
    scanned: dict[str, Any] | None = None
    if local_ok:
        print(
            f"scanning {len(pdfs)} sampled PDFs for scanned pages"
            f" (<{SCAN_MIN_CHARS} chars/page)...",
            file=sys.stderr,
        )
        scanned = scanned_block([data_dir / p["path"] for p in pdfs])
    else:
        print(
            "sampled PDFs not (all) local yet: the scanned-page share is filled in by `fetch`",
            file=sys.stderr,
        )
    manifest = {
        "ticket": 273,
        "dataset": HF_REPO,
        "revision": HF_REVISION,
        "parquet": PARQUET_PATH,
        "parquet_sha256": parquet_sha,
        "parquet_rows": n_rows,
        "seed": args.seed,
        "corpus": (
            f"{len(chosen_docs)}-PDF document-first corpus (ticket 273); the 272 110-PDF "
            "sample is kept as manifest-110.json / questions-110.jsonl"
        ),
        "doc_count": len(chosen_docs),
        "doc_n_target": args.docs,
        "doc_page_budget": args.page_budget,
        "doc_pages_total": docs_pages_total,
        "doc_priority_min_usable": DOC_PRIORITY_MIN_USABLE,
        "doc_selection_rule": (
            f"docs ranked by (usable<{DOC_PRIORITY_MIN_USABLE}, "
            f"int(sha256('mmlongdoc-doc:{args.seed}:<doc_id>'), 16), doc_id); the top "
            f"{args.docs} are taken (usable = answerable, not vision-only questions); "
            f"page cap: while the chosen pages exceed {args.page_budget}, swap the largest "
            "chosen doc for the smallest not-chosen doc (the >=6-usable tier first) when the "
            "swap lowers the total and keeps every evidence capability (text/table/layout/"
            "unanswerable-bearing) the dataset carries; pages: the pinned PDF's real count "
            "where local or fetched, else the max evidence page from the parquet "
            "(per-doc page_source below)"
        ),
        "docs": [
            {
                "doc_id": d,
                "path": f"{DOCS_PREFIX}{d}",
                "pages": pages[d],
                "page_source": page_source[d],
                "questions": stats[d]["n"],
                "usable": stats[d]["usable"],
                "unanswerable": stats[d]["unans"],
                "caps": "".join(sorted(stats[d]["caps"])),
            }
            for d in chosen_docs
        ],
        "sample_rule": (
            "questions sampled ONLY from the chosen documents: qid = q<parquet_row_index:04d>; "
            "within a stratum, rank = "
            f"int(sha256('mmlongdoc-sample:{args.seed}:<qid>'), 16) ascending; the top quota "
            "rows are kept"
        ),
        "split_rule": (
            f"split = 'dev' iff int(sha256('mmlongdoc-split:{args.seed}:<qid>'), 16) % 2 == 0 "
            "else 'test'"
        ),
        "strata_rule": (
            "unanswerable (answer == 'Not answerable') is one stratum with a scaled quota "
            f"(min(pool unanswerable, round(total * {UNANSWERABLE_FRACTION})) unless "
            "--unanswerable is given); the rest are stratified by (evidence, doc_type), "
            "evidence priority text > table > layout; vision-only (Chart/Figure) rows are the "
            "chart_image stratum, flagged out_of_scope_vision: kept in the sample, never in "
            "the headline"
        ),
        "total": total_target,
        "pool_size": len(pool),
        "unanswerable_quota": unans_quota,
        "counts": count_sample(selected),
        "quotas": {"unanswerable": unans_quota},
        "pdfs": pdfs,
        "pdf_count": len(pdfs),
        "scanned_pages": scanned,
        "data_dir": str(data_dir),
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    questions_path = eval_dir / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(
                json.dumps(
                    {
                        "id": r["qid"],
                        "doc_id": r["doc_id"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "answer_format": r["answer_format"],
                        "evidence_pages": r["evidence_pages"],
                        "evidence_sources": r["evidence_sources"],
                        "split": r["split"],
                        "flags": {
                            "out_of_scope_vision": bool(r["out_of_scope_vision"]),
                            "stratum": r["evidence_stratum"],
                            "doc_type": r["doc_type"],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    c = manifest["counts"]
    print(
        f"parquet rows: {n_rows} (paper: 1,082; HF revision: 1,091)  sha256={parquet_sha[:16]}..."
    )
    print(
        f"documents: {len(chosen_docs)} chosen, {docs_pages_total} pages (budget "
        f"{args.page_budget}; "
        + ", ".join(
            f"page_source {src}={n}"
            for src, n in sorted(Counter(page_source[d] for d in chosen_docs).items())
        )
        + ")"
    )
    print(f"question pool of the chosen documents: {len(pool)} (unanswerable {pool_unans})")
    if len(pool) < args.total:
        print(f"the pool is under {args.total}: using all {len(pool)}, no padding")
    print(
        f"sampled: {c['total']} rows  dev={c['by_split']['dev']} test={c['by_split']['test']}  "
        f"unanswerable={c['unanswerable']}  in_scope={c['in_scope']}  "
        f"out_of_scope_vision={c['out_of_scope_vision']}"
    )
    print("by evidence: " + "  ".join(f"{k}={v}" for k, v in c["by_evidence"].items()))
    print("by stratum: " + "  ".join(f"{k}={v}" for k, v in c["by_stratum"].items()))
    share = (scanned or {}).get("share")
    if share is None:
        share_txt = "n/a (fetch first)"
    elif isinstance(share, float):
        share_txt = f"{share:.1%} of the sampled PDFs"
    else:
        share_txt = str(share)
    print(f"scanned pages: {share_txt}")
    print(f"wrote {manifest_path}")
    print(f"wrote {questions_path}")
    return 0


# --- fetch ---------------------------------------------------------------------
def cmd_fetch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).expanduser()
    pdfs: list[dict[str, str]] = manifest["pdfs"]
    for i, p in enumerate(pdfs, start=1):
        local = data_dir / p["path"]
        if local.is_file() and hash_file(local, p["algo"]) == p["hash"]:
            print(f"[{i}/{len(pdfs)}] {p['path']} already verified, skip")
            continue
        print(f"[{i}/{len(pdfs)}] downloading {p['path']} ...", file=sys.stderr)
        download_file(hf_url(p["path"]), local)
        got = hash_file(local, p["algo"])
        if got != p["hash"]:
            local.unlink(missing_ok=True)
            raise RuntimeError(
                f"{p['algo']} mismatch for {p['path']}: want {p['hash'][:16]}... got {got[:16]}..."
            )
        print(f"[{i}/{len(pdfs)}] {p['path']} ok ({local.stat().st_size / 1e6:.1f} MB)")
    print(
        f"scanning {len(pdfs)} PDFs for scanned pages (<{SCAN_MIN_CHARS} chars/page, pypdf)...",
        file=sys.stderr,
    )
    scanned = scanned_block(
        [data_dir / p["path"] for p in pdfs],
        on_pdf=lambda p, t, s: print(f"  {p.name}: {s}/{t} scanned pages"),
    )
    manifest["scanned_pages"] = scanned
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    share = scanned["share"]
    print(
        f"all {len(pdfs)} PDFs verified; scanned pages: {scanned['scanned_pages']}/"
        f"{scanned['total_pages']} = {share:.1%} of the pages in the sampled PDFs"
        if share is not None
        else "all PDFs verified; no pages found"
    )
    print(f"updated {manifest_path}")
    return 0


# --- ingest ---------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).expanduser()
    pdfs: list[dict[str, str]] = manifest["pdfs"]
    for p in pdfs:
        local = data_dir / p["path"]
        if not local.is_file() or hash_file(local, p["algo"]) != p["hash"]:
            raise RuntimeError(f"{p['path']} missing or {p['algo']} mismatch - run `fetch` first")
    corpus_dir = Path(args.corpus_dir)
    db = Path(args.db)
    expected = sorted(p["doc_id"] for p in pdfs)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(x.name for x in corpus_dir.glob("*.pdf"))
    if have != expected:
        for x in corpus_dir.glob("*.pdf"):
            x.unlink()
        for name in expected:
            src = next(data_dir / p["path"] for p in pdfs if p["doc_id"] == name)
            shutil.copy2(src, corpus_dir / name)
        print(f"rebuilt the flat corpus folder {corpus_dir} ({len(expected)} PDFs)")
    if db.exists() and args.force:
        db.unlink()
        db.with_name(db.name + ".fts").unlink(missing_ok=True)
        print("removed the existing DB (--force)")
    # the real ingest (repo src on sys.path, set up in main); folder in, no --ocr (T5)
    from voice_agent.rag.ingest import ingest

    pdf_engine = os.environ.get("VOICE_AGENT_RAG_PDF_ENGINE", "auto")
    report = ingest(corpus_dir, db, workers="auto", ocr_run=False)
    record = {
        "ticket": 273,
        "db": str(db),
        "corpus_dir": str(corpus_dir),
        "ocr": False,
        "ocr_note": "ingest WITHOUT --ocr for this ticket; OCR is ticket T5",
        "pdf_engine": pdf_engine,
        "pdfs": len(expected),
        "files_total": report.files_total,
        "ok": report.ok,
        "failed": report.failed,
        "skipped": report.skipped,
        "chunks": report.chunks,
        "seconds": round(report.seconds, 2),
        "peak_rss_mb": round(report.peak_rss_mb, 1),
    }
    # the record lives next to the DB it describes (out/ is never committed);
    # the 272 record stays at out/mmlongdoc/ingest.json, untouched
    out_dir = db.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingest.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"ingest: {report.ok}/{report.files_total} files ok ({report.failed} failed), "
        f"{report.chunks} chunks, {report.seconds:.1f}s"
    )
    print(f"wrote {out_path}")
    if report.failed or report.ok < len(expected):
        raise RuntimeError(
            f"ingest incomplete: {report.ok}/{report.files_total} ok, {report.failed} failed "
            "- see the files table in the DB for per-file errors"
        )
    return 0


# --- run -------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RuntimeError(f"{path}:{i + 1}: not a JSON line ({exc})") from None
        for key in ("id", "doc_id", "question", "split"):
            if not str(row.get(key) or "").strip():
                raise RuntimeError(f"{path}:{i + 1}: missing {key!r}")
        rows.append(row)
    rows.sort(key=lambda r: str(r["id"]))
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(str(json.loads(line)["id"]))
        except (ValueError, KeyError):
            continue  # a torn tail line: the row is simply re-run
    return done


def record_from_result(row: Mapping[str, Any], res: Any, seconds: float) -> dict[str, Any]:
    """Map a DocResult-shaped lane result onto the run record. The question is asked
    as is, against the whole corpus DB; ``file_matches_doc_id`` records whether the
    cited file is the question's own document (None when nothing was cited)."""
    file_ = getattr(res, "file", None)
    cited = str(file_) if file_ else None
    doc_id = str(row.get("doc_id") or "")
    spoken = getattr(res, "spoken", None)
    return {
        "id": str(row["id"]),
        "doc_id": doc_id,
        "split": str(row.get("split") or ""),
        "question": str(row.get("question") or ""),
        "status": str(getattr(res, "status", "error")),
        "spoken": (str(spoken).strip() or None) if spoken else None,
        "file": cited,
        "file_matches_doc_id": (Path(cited).name == doc_id) if cited else None,
        "quotes": [str(q) for q in (getattr(res, "quotes", None) or [])],
        "cited_pages": [int(x) for x in (getattr(res, "cited_pages", None) or [])],
        "quote_files": [str(f) for f in (getattr(res, "quote_files", None) or [])],
        "calls": int(getattr(res, "calls", 0) or 0),
        "seconds": round(seconds, 3),
    }


async def run_questions(
    rows: Sequence[Mapping[str, Any]],
    ask: Callable[[str, Mapping[str, Any]], Awaitable[Any]],
    out_path: Path,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Answer the pending rows through ``ask(question, row)`` (the real lane),
    appending one JSON line per row. Resumable: ids already in ``out_path`` are
    skipped. Returns (new records, skipped count)."""
    done = read_done_ids(out_path)
    pending = [r for r in rows if str(r["id"]) not in done]
    skipped = len(rows) - len(pending)
    new: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for row in pending:
            t0 = time.perf_counter()
            res = await ask(str(row["question"]), row)
            rec = record_from_result(row, res, time.perf_counter() - t0)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            new.append(rec)
            if on_row is not None:
                on_row(rec)
    return new, skipped


def make_doc_request(
    transport: Any, model: str, se: Any, failures: list[str]
) -> Callable[..., Awaitable[str]]:
    """The document lane's Qwen request: the sufficiency_eval shared JSON payload,
    the JSON-only suffix on the system prompt (the relay is OpenAI-format, so the
    schema is enforced the same way as there)."""

    async def doc_request(system: str, content: str, schema: Any = None) -> str:
        payload = se.relay_payload(model, system + se.JSON_ONLY_SUFFIX, content)
        try:
            data = await asyncio.to_thread(transport.post, payload)
        except Exception as exc:
            failures.append(f"doc: {exc}")
            raise
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            failures.append("doc: the relay returned no message content")
            raise RuntimeError("the relay returned no message content")
        return text

    return doc_request


GEMINI_USD_PER_M_IN = 0.30  # conservative Flash-Lite-class prices; the cap uses these
GEMINI_USD_PER_M_OUT = 2.50


class SpendCapError(RuntimeError):
    """The estimated Gemini spend passed the owner's cap; the run stops (resumable)."""


async def make_gemini_request(cap_usd: float) -> Callable[..., Awaitable[str]]:
    """The production lane model (Gemini via `gemini_json_request`, key from ~/.secrets, never
    printed). Spend is estimated from character counts (4 per token) at conservative prices and
    the run raises SpendCapError before a call that would pass `cap_usd`."""
    from voice_pipeline.ask_documents import gemini_json_request

    inner = await gemini_json_request()
    spent = {"usd": 0.0, "calls": 0}

    async def request(system: str, content: str, schema: Any = None) -> str:
        est_in = (len(system) + len(content)) / 4
        if spent["usd"] + est_in * GEMINI_USD_PER_M_IN / 1e6 > cap_usd:
            raise SpendCapError(f"estimated Gemini spend ${spent['usd']:.3f} at the ${cap_usd} cap")
        reply = (
            await inner(system, content, schema)
            if schema is not None
            else await inner(system, content)
        )
        spent["calls"] += 1
        spent["usd"] += (
            est_in * GEMINI_USD_PER_M_IN + (len(reply) / 4) * GEMINI_USD_PER_M_OUT
        ) / 1e6
        print(f"    gemini calls={spent['calls']} est_spend=${spent['usd']:.4f}", file=sys.stderr)
        return reply

    return request


def relay_preflight(transport: Any, model: str) -> None:
    """One tiny free call before the run: a dead relay aborts loudly up front
    instead of scoring every row as status 'error' (the lane swallows request
    exceptions by design)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Reply with the JSON object {"ok": true} only.'}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    transport.post(payload)


async def cmd_run(args: argparse.Namespace) -> int:
    se = load_sufficiency_eval()
    now = se.manila_now(args.today)
    if now is None:
        print(f"--today must be a YYYY-MM-DD date (got {args.today!r})", file=sys.stderr)
        return 2
    questions = load_questions(DEFAULT_QUESTIONS)
    rows = [r for r in questions if r["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(f"no rows for split={args.split} limit={args.limit}", file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else OUT_DIR / f"run-{args.tag}.jsonl"
    done = read_done_ids(out_path)
    pending_ids = [str(r["id"]) for r in rows if str(r["id"]) not in done]
    if not pending_ids:
        print(f"all {len(rows)} rows of split={args.split} already in {out_path}; nothing to do")
        return 0
    db = Path(args.db)
    corpus_dir = Path(args.corpus_dir)
    try:
        from voice_pipeline.ask_documents import ask_documents
        from voice_pipeline.doc_answer import passages_with_meta

        from voice_agent.rag.dbindex import DbIndex
    except ImportError as exc:
        print(
            f"this run needs the pipeline venv and the repo src on sys.path: {exc}", file=sys.stderr
        )
        return 2
    index = DbIndex(db, raw_root=corpus_dir)
    file_names = se.load_file_names(db)
    cards = se.load_cards(db)

    async def context_fn(question: str) -> list[dict[str, Any]]:
        return passages_with_meta(await asyncio.to_thread(index.file_context, question))

    failures: list[str] = []
    if args.lane_backend == "gemini":
        doc_request = await make_gemini_request(args.spend_cap_usd)
    else:
        transport = se.Relay(args.base_url, read_key(), args.model)
        print("relay preflight (one tiny free call)...", file=sys.stderr)
        relay_preflight(transport, args.model)
        doc_request = make_doc_request(transport, args.model, se, failures)

    async def ask(question: str, row: Mapping[str, Any]) -> Any:
        # the real document lane, docs-only: no web, no supplement, no fallback
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
            budget_s=args.budget_s,
            allow_fallback=False,
        )

    consecutive_errors = 0
    progress = {"done": 0}

    def on_row(rec: dict[str, Any]) -> None:
        nonlocal consecutive_errors
        progress["done"] += 1
        if rec["status"] == "error":
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
            raise RunAbortedError(
                f"{CONSECUTIVE_ERROR_LIMIT} consecutive 'error' rows "
                f"(last id {rec['id']}); the relay is likely down - "
                "aborting, the run file is resumable"
            )
        status = f"{rec['status']:<12}"
        print(
            f"  [{progress['done']}/{len(pending_ids)}] {rec['id']} status={status} "
            f"{rec['seconds']:.1f}s spoken={bool(rec['spoken'])}",
            file=sys.stderr,
        )

    print(
        f"running {len(pending_ids)} questions (split={args.split}, tag={args.tag}) "
        f"-> {out_path} ({len(done)} already done)",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        new, skipped = await run_questions(rows, ask, out_path, on_row=on_row)
    finally:
        index.close()
    total_s = time.perf_counter() - t0
    status_counts = Counter(r["status"] for r in new)
    counts_txt = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    print(f"done: {len(new)} new + {skipped} skipped in {total_s:.0f}s ({counts_txt})")
    return 0


# --- score -----------------------------------------------------------------------
def extract_payload(model: str, question: str, spoken: str) -> dict[str, Any]:
    """The official extractor prompt (verbatim) + the official question/analysis
    layout, adapted to one JSON user message for the relay (temperature 0, the
    official max_tokens=256)."""
    content = (
        OFFICIAL_EXTRACTOR_PROMPT
        + f"\n\nQuestion:{question}\nAnalysis:{spoken}\n"
        + EXTRACT_JSON_INSTRUCTION
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def extract_from_spoken(transport: Any, model: str, question: str, spoken: str) -> tuple[Any, str]:
    data = transport.post(extract_payload(model, question, spoken))
    choices = data.get("choices") or []
    message = choices[0].get("message") if choices else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("the extractor relay call returned no message content")
    try:
        reply = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"the extractor returned non-JSON: {text[:200]}") from exc
    if not isinstance(reply, dict):
        raise RuntimeError(f"the extractor returned a non-object: {text[:200]}")
    ans = reply.get("extracted_answer")
    fmt = str(reply.get("answer_format") or "String")
    if ans is None:
        ans = NOT_ANSWERABLE
    elif not isinstance(ans, (str, list)):
        ans = str(ans)
    return ans, fmt


def extract_for_row(
    row: Mapping[str, Any], transport: Any | None, model: str
) -> tuple[Any, str, str]:
    """(extracted, extracted_format, extractor). A row the lane did not accept has
    nothing spoken in docs-only mode: the honest extraction is 'Not answerable',
    with no relay call. Accepted rows go through the Qwen extractor."""
    if str(row.get("status")) == "accepted" and row.get("spoken"):
        if transport is None:
            raise RuntimeError(
                f"row {row.get('id')!r} was accepted and spoken but has no transport"
            )
        ans, fmt = extract_from_spoken(
            transport, model, str(row.get("question") or ""), str(row["spoken"])
        )
        return ans, fmt, "qwen"
    return NOT_ANSWERABLE, "String", f"lane:{row.get('status')}"


def score_row(
    gold: Any, extracted: Any, answer_format: str, eval_score_fn: Callable[..., float]
) -> float:
    """The vendored official rule, called exactly as the official pipeline calls it
    (gold, pred, answer_type). The List branch prints debug output; it is silenced,
    not removed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return float(eval_score_fn(gold, extracted, answer_format))


def join_run_with_questions(
    run_rows: Sequence[Mapping[str, Any]], question_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(r["id"]): r for r in question_rows}
    out: list[dict[str, Any]] = []
    for r in run_rows:
        meta = by_id.get(str(r["id"]))
        if meta is None:
            raise RuntimeError(
                f"run row {r.get('id')!r} is not in {DEFAULT_QUESTIONS} - did the manifest change?"
            )
        flags = meta.get("flags") or {}
        out.append(
            {
                "id": str(r["id"]),
                "doc_id": str(meta.get("doc_id") or ""),
                "question": str(r.get("question") or meta.get("question") or ""),
                "answer": str(meta.get("answer") or ""),
                "answer_format": str(meta.get("answer_format") or ""),
                "evidence_sources": list(meta.get("evidence_sources") or []),
                "split": str(meta.get("split") or r.get("split") or ""),
                "out_of_scope_vision": bool(flags.get("out_of_scope_vision")),
                "stratum": str(flags.get("stratum") or ""),
                "doc_type": str(flags.get("doc_type") or ""),
                "status": str(r.get("status") or ""),
                "spoken": r.get("spoken"),
                "file": r.get("file"),
                "file_matches_doc_id": r.get("file_matches_doc_id"),
            }
        )
    return out


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    transport: Any | None,
    model: str,
    eval_score_fn: Callable[..., float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        gold = str(row.get("answer") if "answer" in row else row.get("gold") or "")
        extracted, fmt, extractor = extract_for_row(row, transport, model)
        score = score_row(gold, extracted, str(row.get("answer_format") or "Str"), eval_score_fn)
        rec = dict(row)
        rec.update(
            {
                "gold": gold,
                "extracted": extracted,
                "extracted_format": fmt,
                "extractor": extractor,
                "score": score,
                "unanswerable": is_unanswerable(gold),
            }
        )
        out.append(rec)
    return out


def _acc(scores: Sequence[float]) -> float | None:
    return round(sum(scores) / len(scores), 4) if scores else None


def aggregate_scores(
    rows: Sequence[Mapping[str, Any]], acc_f1_fn: Callable[..., Any]
) -> dict[str, Any]:
    """Headline accuracy on the in-scope answerable rows (text/table/layout),
    unanswerable accuracy, the OFFICIAL F1 over the in-scope rows (the official
    F1 is a whole-set metric: its precision term counts "Not answerable"
    predictions, so it must include both answerable and unanswerable rows -
    computed on unanswerable rows alone it is always 0 by construction), by
    answer_format / evidence source / split, the cited-file match, and the
    out_of_scope_vision rows reported separately - never in the headline."""

    def official_samples(rs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{"answer": r["gold"], "pred": r["extracted"], "score": r["score"]} for r in rs]

    vision = [r for r in rows if r["out_of_scope_vision"]]
    core = [r for r in rows if not r["out_of_scope_vision"]]
    headline = [r for r in core if not r["unanswerable"]]
    unans = [r for r in core if r["unanswerable"]]
    core_acc, core_f1 = acc_f1_fn(official_samples(core))
    overall_acc, overall_f1 = acc_f1_fn(official_samples(rows))
    by_format: dict[str, dict[str, Any]] = {}
    for fmt in sorted({str(r["answer_format"]) for r in core}):
        rs = [r for r in core if str(r["answer_format"]) == fmt]
        by_format[fmt] = {"n": len(rs), "acc": _acc([r["score"] for r in rs])}
    by_source: dict[str, dict[str, Any]] = {}
    for src in sorted({s for r in core for s in r["evidence_sources"]}):
        rs = [r for r in core if src in r["evidence_sources"]]
        by_source[src] = {"n": len(rs), "acc": _acc([r["score"] for r in rs])}
    by_split: dict[str, dict[str, Any]] = {}
    for sp in ("dev", "test"):
        rs = [r for r in core if str(r["split"]) == sp]
        if rs:
            by_split[sp] = {"n": len(rs), "acc": _acc([r["score"] for r in rs])}
    file_rows = [r for r in core if r.get("file_matches_doc_id") is not None]
    return {
        "n_rows": len(rows),
        "headline": {
            "n": len(headline),
            "acc": _acc([r["score"] for r in headline]),
            "note": "in-scope answerable rows (text/table/layout evidence); unanswerable"
            "and vision rows excluded",
        },
        "core": {
            "n": len(core),
            "acc": round(float(core_acc), 4),
            "f1": round(float(core_f1), 4),
            "note": (
                "in-scope rows (answerable + unanswerable); the official F1 over this "
                "set is the unanswerable-aware F1 (its precision counts 'Not answerable' preds)"
            ),
        },
        "unanswerable": {
            "n": len(unans),
            "acc": _acc([r["score"] for r in unans]),
        },
        "official_overall": {
            "n": len(rows),
            "acc": round(float(overall_acc), 4),
            "f1": round(float(overall_f1), 4),
            "note": "official eval_acc_and_f1 over every scored row, incl. the"
            "out-of-scope vision rows",
        },
        "by_answer_format": by_format,
        "by_evidence_source": by_source,
        "by_split": by_split,
        "cited_file_match": {
            "n": len(file_rows),
            "matches": sum(1 for r in file_rows if r["file_matches_doc_id"]),
        },
        "out_of_scope_vision": {
            "n": len(vision),
            "acc": _acc([r["score"] for r in vision]),
            "note": "chart/image-only evidence; reported separately, never in the headline",
        },
        "extractors": dict(sorted(Counter(str(r.get("extractor", "?")) for r in rows).items())),
    }


def render_score_table(report: Mapping[str, Any]) -> str:
    a = report["aggregate"]
    tag = str(report.get("tag") or "")
    lines: list[str] = []
    lines.append(f"# MMLongBench-Doc bench score - tag {tag}")
    lines.append("")
    lines.append(
        f"rows: {a['n_rows']} (answerer: local Qwen; extractor: local Qwen;"
        f" rules: official, unchanged)"
    )
    lines.append("")
    lines.append("| metric | n | acc |")
    lines.append("| :--- | ---: | ---: |")
    lines.append(
        f"| **headline: in-scope answerable** | {a['headline']['n']} | {a['headline']['acc']} |"
    )
    lines.append(f"| in-scope incl. unanswerable | {a['core']['n']} | {a['core']['acc']} |")
    lines.append(
        f"| official overall (incl. vision) | {a['official_overall']['n']} "
        f"| {a['official_overall']['acc']} |"
    )
    for fmt, m in a["by_answer_format"].items():
        lines.append(f"| by answer_format: {fmt} | {m['n']} | {m['acc']} |")
    for src, m in a["by_evidence_source"].items():
        lines.append(f"| by evidence source: {src} | {m['n']} | {m['acc']} |")
    for sp, m in a["by_split"].items():
        lines.append(f"| by split: {sp} | {m['n']} | {m['acc']} |")
    lines.append("")
    un = a["unanswerable"]
    lines.append(f"unanswerable: acc {un['acc']} (n={un['n']})")
    lines.append(f"official F1 over the in-scope rows (unanswerable-aware): {a['core']['f1']}")
    fm = a["cited_file_match"]
    lines.append(f"cited file == question's doc_id: {fm['matches']}/{fm['n']}")
    vis = a["out_of_scope_vision"]
    lines.append("")
    lines.append(
        f"out_of_scope_vision (chart/image-only; NOT in the headline): "
        f"n={vis['n']}, acc {vis['acc']}"
    )
    lines.append("")
    lines.append("extractor: " + ", ".join(f"{k}={v}" for k, v in a["extractors"].items()))
    return "\n".join(lines)


def cmd_score(args: argparse.Namespace) -> int:
    se = load_sufficiency_eval()
    run_path = Path(args.run) if args.run else OUT_DIR / f"run-{args.tag}.jsonl"
    if not run_path.is_file():
        print(f"run file not found: {run_path} (run `run --tag {args.tag}` first)", file=sys.stderr)
        return 2
    run_rows: list[dict[str, Any]] = []
    for i, line in enumerate(run_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            run_rows.append(json.loads(line))
        except ValueError as exc:
            raise RuntimeError(f"{run_path}:{i + 1}: not a JSON line ({exc})") from None
    if not run_rows:
        print(f"no rows in {run_path}", file=sys.stderr)
        return 2
    questions = load_questions(DEFAULT_QUESTIONS)
    joined = join_run_with_questions(run_rows, questions)
    official = load_official()
    needs_relay = any(r["status"] == "accepted" and r.get("spoken") for r in joined)
    transport: Any | None = None
    if needs_relay:
        transport = se.Relay(args.base_url, read_key(), args.model)
        relay_preflight(transport, args.model)
    n_spoken = sum(1 for r in joined if r["status"] == "accepted" and r.get("spoken"))
    print(
        f"scoring {len(joined)} rows ({n_spoken} spoken -> Qwen extraction; the rest "
        "'Not answerable' from the lane status)",
        file=sys.stderr,
    )
    scored = score_rows(joined, transport, args.model, official.eval_score)
    aggregate = aggregate_scores(scored, official.eval_acc_and_f1)
    report = {
        "ticket": 273,
        "tag": args.tag,
        "run": str(run_path),
        "model": args.model,
        "official_rules": "scripts/mmlongdoc_eval_score.py (vendored verbatim, Apache-2.0)",
        "extractor": "Qwen, the official prompt adapted to JSON (temperature 0)",
        "aggregate": aggregate,
        "rows": scored,
    }
    out_path = Path(args.out) if args.out else OUT_DIR / f"score-{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = render_score_table(report)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    return 0


# --- calibration-sheet -------------------------------------------------------------
def select_calibration(
    rows: Sequence[Mapping[str, Any]], n: int = CALIBRATION_N, seed: str = SEED
) -> list[dict[str, Any]]:
    """n stratified seeded rows for the hand audit: proportional over the strata
    (minimum 1), ranked by a dedicated salt so the audit set is stable per seed."""
    population: Counter[str] = Counter(str(r.get("stratum") or "?") for r in rows)
    quotas = allocate_quotas(population, min(n, len(rows)))
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("stratum") or "?"), []).append(r)
    selected: list[dict[str, Any]] = []
    for stratum, group in groups.items():
        take = quotas.get(stratum, 0)
        if take <= 0:
            continue
        ranked = sorted(
            group,
            key=lambda r: (sha_rank(f"mmlongdoc-calib:{seed}", str(r["id"])), str(r["id"])),
        )
        selected.extend(dict(r) for r in ranked[:take])
    selected.sort(key=lambda r: str(r["id"]))
    return selected


def cmd_calibration(args: argparse.Namespace) -> int:
    score_path = Path(args.score) if args.score else OUT_DIR / f"score-{args.tag}.json"
    if not score_path.is_file():
        print(
            f"score file not found: {score_path} (run `score --tag {args.tag}` first)",
            file=sys.stderr,
        )
        return 2
    report = json.loads(score_path.read_text(encoding="utf-8"))
    rows = report.get("rows") or []
    if not rows:
        print(f"no rows in {score_path}", file=sys.stderr)
        return 2
    selected = select_calibration(rows, n=args.n, seed=args.seed)
    lines: list[str] = []
    lines.append(f"# MMLongBench-Doc calibration sheet - {len(selected)} rows (ticket 273)")
    lines.append("")
    lines.append(
        "For the orchestrator's hand audit (the ticket-269 calibration gate: Qwen vs the "
        "official extraction, agreement >= 90% / kappa >= 0.80). The verdict column is the "
        "vendored official rule applied to Qwen's extracted answer - shown as is, not graded here."
    )
    lines.append("")
    for r in selected:
        vision = " [VISION-ONLY]" if r.get("out_of_scope_vision") else ""
        lines.append(
            f"## {r['id']}  [{r.get('split')}] stratum={r.get('stratum')}{vision} "
            f"format={r.get('answer_format')} verdict(score)={r.get('score')}"
        )
        lines.append(f"- question: {r.get('question')}")
        lines.append(f"- gold: {r.get('gold')}")
        spoken = r.get("spoken") or f"(none: lane status {r.get('status')})"
        lines.append(f"- spoken: {spoken}")
        extracted = r.get("extracted")
        lines.append(
            f"- extracted (Qwen, {r.get('extractor')}): {extracted!r} "
            f"[format: {r.get('extracted_format')}]"
        )
        lines.append("")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"calibration sheet: {len(selected)} rows -> {out_path}")
    return 0


# --- CLI ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="bench_mmlongdoc",
        description="MMLongBench-Doc offline bench (ticket 272):",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="write the manifest and the sampled questions")
    s.add_argument("--seed", default=SEED, help=f"the sha256 salt (default {SEED})")
    s.add_argument(
        "--docs", type=int, default=DOC_N, help=f"documents in the corpus (default {DOC_N})"
    )
    s.add_argument(
        "--page-budget",
        type=int,
        default=DOC_PAGE_BUDGET,
        help=f"page cap for the corpus (default {DOC_PAGE_BUDGET})",
    )
    s.add_argument("--total", type=int, default=TOTAL, help=f"total sample size (default {TOTAL})")
    s.add_argument(
        "--unanswerable",
        type=int,
        default=None,
        help=(
            "unanswerable quota (default: min(pool unanswerable, round(total * "
            f"{UNANSWERABLE_FRACTION})))"
        ),
    )
    s.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR), help="raw data root (NOT in the repo)"
    )
    s.add_argument(
        "--eval-dir", default=str(EVAL_DIR), help="manifest/questions output dir (committed)"
    )

    f = sub.add_parser(
        "fetch", help="download and hash-verify the manifest's PDFs, record the scanned share"
    )
    f.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    f.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    g = sub.add_parser("ingest", help="one corpus DB for the sampled PDFs (flat folder, no --ocr)")
    g.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    g.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    g.add_argument("--db", default=str(DEFAULT_DB), help="corpus DB path (out/, never committed)")
    g.add_argument(
        "--corpus-dir", default=str(DEFAULT_CORPUS_DIR), help="flat PDF folder the DB is built from"
    )
    g.add_argument(
        "--force", action="store_true", help="remove the existing DB first (clean rebuild)"
    )

    r = sub.add_parser("run", help="answer the split's questions through ask_documents (docs-only)")
    r.add_argument("--split", choices=("dev", "test"), required=True)
    r.add_argument("--limit", type=int, default=None, help="first N rows of the split (by id)")
    r.add_argument("--tag", required=True, help="run tag (out/mmlongdoc/run-<tag>.jsonl)")
    r.add_argument("--db", default=str(DEFAULT_DB))
    r.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    r.add_argument(
        "--out",
        default=None,
        help="explicit run jsonl path (default out/mmlongdoc/run-<tag>.jsonl)",
    )
    r.add_argument(
        "--today", default=DEFAULT_TODAY, help="fixed date the lane's clock sees (YYYY-MM-DD)"
    )
    r.add_argument(
        "--budget-s", type=float, default=DOC_BUDGET_S, help="per-question lane budget in seconds"
    )
    r.add_argument("--base-url", default=base_url_default(), help="the Qwen relay base URL")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument(
        "--lane-backend",
        choices=("qwen", "gemini"),
        default="qwen",
        help="who answers the document lane; gemini is PAID (needs the owner's spend approval)",
    )
    r.add_argument(
        "--spend-cap-usd", type=float, default=0.30, help="stop a gemini run at this estimate"
    )

    c = sub.add_parser("score", help="Qwen extraction + the official rules; aggregate table")
    c.add_argument("--tag", required=True)
    c.add_argument(
        "--run", default=None, help="explicit run jsonl (default out/mmlongdoc/run-<tag>.jsonl)"
    )
    c.add_argument(
        "--out", default=None, help="explicit score json (default out/mmlongdoc/score-<tag>.json)"
    )
    c.add_argument("--base-url", default=base_url_default())
    c.add_argument("--model", default=DEFAULT_MODEL)

    k = sub.add_parser("calibration-sheet", help="50 stratified seeded rows for the hand audit")
    k.add_argument("--tag", required=True, help="the scored run's tag")
    k.add_argument(
        "--score", default=None, help="explicit score json (default out/mmlongdoc/score-<tag>.json)"
    )
    k.add_argument(
        "--n", type=int, default=CALIBRATION_N, help=f"rows in the sheet (default {CALIBRATION_N})"
    )
    k.add_argument("--seed", default=SEED)
    k.add_argument("--out", default=str(OUT_DIR / f"calibration-{CALIBRATION_N}.md"))

    return ap.parse_args(argv)


def base_url_default() -> str:
    """The relay base URL: JUDGE_BASE_URL in the environment, else the local relay."""
    return os.environ.get("JUDGE_BASE_URL", DEFAULT_BASE_URL)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    src_dir = str(REPO_ROOT / "src")
    pipe_dir = str(REPO_ROOT / "pipeline" / "src")
    for d in (src_dir, pipe_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    handlers = {
        "sample": cmd_sample,
        "fetch": cmd_fetch,
        "ingest": cmd_ingest,
        "score": cmd_score,
        "calibration-sheet": cmd_calibration,
    }
    try:
        if args.cmd == "run":
            return asyncio.run(cmd_run(args))
        return handlers[args.cmd](args)
    except RunAbortedError as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
