# ME4 specification and progress assessment

Written 2026-11-20 on branch `local/agent-assessment`. Part 1 is the specification as given by the owner (verbatim). Part 2 breaks it into checkable requirements and states, from the code and the measurements in this repo, how far each is met.

## 1. The specification (verbatim)

> This ME is basically the same as ME3. The only difference is it is going to base its answers on facts that can only be found in reference data or files. On the day of checking, you will be given a zip folder with documents of various forms (office, md, pdf, csv, etc). Your agent must be able to answer the query based on the facts stored in the documents. The answer may come from one or more documents plus other external sources such as the web.

## 2. Requirements and status

Legend: **met** = built and measured; **partial** = built, with a known gap; **gap** = not built.

| # | Requirement (derived from the spec) | Status | Evidence / what is missing |
|---|---|---|---|
| R1 | Same agent as ME3 (voice and text, tools, web) | **met** | The document route sits in front of the ME3 turn path (`pipeline/src/voice_pipeline/doc_route.py`); non-document turns fall through to the ME3 route unchanged. |
| R2 | Input arrives as **a zip folder** | **gap** | Nothing unzips an archive. The docs server reads a plain folder (`--corpus <dir>`); `zipguard` only protects OOXML members. On the day someone must unzip by hand into the corpus folder. Cheap fix: accept a `.zip` (extract with member-count, size, ratio and path-traversal limits into the corpus folder). |
| R3 | "Documents of various forms (office, md, pdf, csv, etc)" | **partial** | Supported: md, txt, csv, pdf, docx, pptx, xlsx, html, epub (`docs/RAG-PI.md`). Not supported: `.doc`/`.ppt`/`.xls`, `.rtf`, `.odt/.ods/.odp`, json/xml/yaml/log, images and scanned PDFs (no OCR). An unsupported file is reported as a failure, never silently dropped. The "etc" is the risk: add cheap plain-text types (json, xml, yaml, tsv, log, rtf) and decide on OCR. |
| R4 | Answers are grounded in facts only found in the documents | **met, with a known weakness** | Verbatim-quote check (`doc_answer.judge`), abstention ("The documents don't say"), implicit gate, prompt-injection guards. Weakness D2: the quote check proves a quote exists, not that it answers (3 of 4 unanswerable golden questions are accepted by the check alone; abstention then depends on the model's own `NONE`). |
| R5 | Answers can combine **more than one document** | **gap** | `DocAnswer` carries one quote and one passage. Retrieval returns several passages (up to 4 per file under the cap), but the answer is a single cited quote, so questions needing two files, a sum across rows, or a comparison are answered from one passage or not at all. Conflicts between files (D6) are not detected. |
| R6 | Answers can combine documents **plus external sources such as the web** | **partial** | Web is reachable only as a fallback: when the documents do not say, the agent offers a web lookup and a "yes" goes to the grounded Gemini route. Pairing a document fact with a web fact in one answer (D5) is not built; Tavily is not configured. |
| R7 | Retrieval quality is good enough to find the right passage | **met on our set, calibrated externally** | 112-question mixed external set: 92/112 hits at k=4 (FTS5), 98/112 hybrid; lexical stack matches the published BM25 on BEIR SciFact, NFCorpus, FiQA, SCIDOCS (E11 in `docs/RAG-EXPERIMENTS.md`). Weakest: csv recall 0.50 and wide spreadsheets (chunking), PDF answers ranked low inside long files. |
| R8 | Runs lean on the Raspberry Pi 4 | **met for retrieval, open for latency** | Pi run: identical retrieval to x86, 199 MiB, 4x slower overall (`docs/RAG-PI-RESULTS.md`). Open: minute-scale first index for big PDFs (pypdfium2 added, Pi timing not yet measured), and the critic findings below. |
| R9 | A live end-to-end evaluation with real model answers | **gap** | All answer-side numbers are offline (scripted model replies); the live benchmark is not approved (spend cap $0.50, about $0.26 used). Real answer accuracy on the golden set has never been measured. |
| R10 | Behaviour on an unseen corpus on the day (nothing tuned to it) | **partial** | The test set was built from public mixed documents and a clean fixture set, not tuned per file; the agent has no per-corpus configuration. Not rehearsed: a cold start on a fresh unzipped corpus with timing, which the work below (parse cache, background indexer) targets. |

## 3. Reliability and latency work since the first Pi run (from the external critic)

Source: `.scratch/rag-markdown/research/critic-opus.md` (Opus critic, read-only; items below marked verified were reproduced by the orchestrator or by the tickets' own tests).

| Finding | State |
|---|---|
| pdfium left U+FFFE inside words (broke search and quote checks) | **fixed**, pushed (`645e179`), regression test |
| every restart re-parsed every PDF/DOCX/CSV | **fixed** (parse cache for legacy formats, ticket 21, uncommitted): second start 14.4 s -> 0.45 s on two IRS PDFs |
| `search_docs`/`docs_status` refreshed inline and blocked the server, startup waited for the first parse | **fixed on the docs-server side** (ticket 22, uncommitted): while a 126-page PDF parsed, `search_docs` max 52 ms, p95 42 ms; the PDF became searchable 14 s after copy (pypdf engine, x86). Pipeline side (timeouts on `docs_status`, "still reading" from the new `indexing`/`parsing` fields) is ticket 23, not started. |
| every non-fast turn with >= 3 content words waits for a document search first | open |
| wide CSVs chunk badly (7,831 chunks from 538 KB, row label lost) | open |
| ASCII-only tokenizer (accents, ligatures) | open |
| agent benchmark records no timing; no load, thermal, SD-card, churn or restart tests | open |

## 4. Gaps ranked for the day of checking

Ranked by (probability the checker hits it) x (damage):

1. **R5/R6 multi-source answers.** The spec says it explicitly ("one or more documents plus ... the web"). Today: single-quote answers. Build a multi-passage answer path (several cited quotes, each verified by `judge`), a "combine with web" mode, and cross-document conflict notes.
2. **R2 zip input.** A checker will hand over a zip. Add safe archive extraction (limits as `zipguard`) so the agent can be pointed at the zip itself.
3. **R9 live evaluation.** Needs the owner's approval for a small live spend; without it real answer quality is unmeasured. A 30-question live run on the golden set costs a fraction of the remaining cap.
4. **R3 coverage of "etc".** Cheap plain-text formats first; then decide on OCR (not Pi-lean).
5. **Latency under a cold unzipped corpus** (tickets 21-23, CSV chunking, pdfium on the Pi).
6. **D2 abstention strength** (answer-relevance check beyond word overlap).

## 5. Day-of checklist (proposed)

1. Unzip (or point at) the corpus; first index must not block the server (ticket 22) and should finish in under a minute for a typical folder (re-measure on the Pi).
2. Check `docs_status` for failures and read any unsupported-file names aloud/in text so nothing is silently ignored.
3. Run 10 spot questions covering: a table cell, a spreadsheet row, a slide, a PDF paragraph, a two-document question, a question that needs the web, and an unanswerable one.
