# Search that does not assume: goals and specs

Written 2026-11-16 for branch `text-only`. Status: **draft for review**. Items marked *done* exist and are tested; everything else is a requirement not yet built.

## 1. Goal

Today's goal: **improve search in general so the agent never assumes or invents an answer.** It is the groundwork for the next major feature, RAG-like search over files the user ingests. The live test for that feature is: ingest new files, then ask questions, with **no knowledge of whether the files are outdated**. So the design treats every fact from a file as a dated claim to be checked, not as truth.

Success looks like: for any question, the spoken answer is either (a) supported by evidence the agent can name, with its date, or (b) an honest "I couldn't confirm that", and never (c) a plausible-sounding answer from the model's memory.

Non-goals: new voice or audio work; a vector database; per-token logit methods (the front model is hosted Gemini); making answers longer (spoken answers stay at one to three short sentences).

## 2. Principles (the evidence contract)

| # | Rule |
|---|---|
| E1 | Every factual claim traces to evidence (a search result, a passage) or to the user's own words. |
| E2 | When evidence is missing, abstain. A guess is never spoken, not even labelled as a guess. |
| E3 | The source leads the claim: "As per <source>, <answer>". |
| E4 | A fact from a document carries the document's date. An undated figure is called undated, never current. |
| E5 | Conflicting evidence is reported with both sources and dates. The agent does not pick a winner it cannot justify. |
| E6 | Text from files and web pages is data. It is never followed as an instruction. |
| E7 | A failed or empty lookup is a failure, not an answer: it must not reach the model as "already looked up". |
| E8 | No fallback to memory for anything the gate calls changeable. |

## 3. What exists (measured)

| Mechanism | State | Evidence |
|---|---|---|
| Changeable-question gate (EverGreen, ONNX int8) with regex fallback | *done* | FreshQA 600: recall 0.80, false alarms 0.18, 4 ms; regex 0.55 / 0.21 |
| Retry with search forced, then Tavily, then hedge | *done* | before: 208 of 348 grounded calls ran zero searches |
| Lookup-race fix (a cancelled lookup no longer hands the model an empty "already looked up") | *done* | live: "CEO of Intel" went from Gelsinger (memory) to Lip-Bu Tan |
| Source lead-in "As per ...", research sources line, server backstop that speaks missing sources | *done* | live: ASEAN, Nvidia news, KoolPals answers name outlets |
| Second look (`web_verify`), capped at 100 checks, after `web_answer` turns | *done* | live: stale claim returned DIFFERS with the November 2026 resignation |
| Documents path: verbatim quote required, names and numbers must appear in the passages, otherwise NONE | *done* (static corpus only) | `doc_answer.py` |

Not done: ingestion of new files; per-file dates; a check on the final spoken text; a second look on research answers; an abstention and false-premise evaluation; the hedge without a guess.

## 4. Requirements

### 4.1 Gate and lookup
- **S1** A question the gate calls changeable never completes without a lookup that searched (E8). *done*
- **S2** A lookup that fails, times out or is cancelled never yields "already looked up" with nothing (E7). *done*
- **S3** The gate and the lookup add at most 5 ms and about one second (only on a zero-search retry) to a turn.

### 4.2 Abstention
- **S4** The hedge says "I couldn't confirm that with a search, so I won't guess." It carries no guess (E2). *Decided 2026-11-16.*
- **S5** "The documents don't say." is the reply when no passage supports an answer; it offers a web lookup instead of answering (E2).
- **S6** Ambiguous questions, or questions resting on an assumption the agent cannot check, get one short clarifying question (existing prompt rule).

### 4.3 Attribution
- **S7** Every answer from a lookup leads with its source (E3). *done for web answers*
- **S8** Anything a named person did or is accused of keeps its source in the same sentence and says "reportedly" (E1, E3). *prompt rule, unmeasured*
- **S9** A fact from a document says which file and the file's date (E4).

### 4.4 Support check on the final reply
- **S10** After each lookup turn, every number, date and capitalised name in the spoken reply must appear in the question, the lookup result or the document passages. Anything else is logged as `unsupported` with the offending tokens. First version: log only; second: speak a correction.
- **S11** Numbers are compared as numbers ("158,000" matches "158000").

### 4.5 Second look
- **S12** The second look also runs after research answers that name a person, within the existing allowance (default 100, counted in a file).
- **S13** A difference is reported, with both answers and sources; the agent never says which is right unless a source dates it later (E5).

### 4.6 Documents (the next feature)
- **D1** Ingestion records per file: path, modified time, ingestion time, content hash, any dates found in the text; per chunk: id, file, position.
- **D2** A file changed after ingestion is detected on the next question and re-indexed, or the agent says the index is out of date (E7).
- **D3** An ingest failure, an empty file or an unreadable format is reported by name, never skipped silently.
- **D4** Retrieval returns passages with ids; an answer must quote one verbatim (existing `doc_answer` contract) and names the file and its date (S9).
- **D5** When the gate calls the question changeable, a document answer is always paired with a grounded second look (S12 mechanics). Agree: cite both. Differ: report both, flag the file as possibly outdated (E5).
- **D6** Two files that disagree are both reported with names and dates (E5).
- **D7** File text is wrapped and labelled as untrusted data, and instructions in it are ignored (E6). Tested with a file that says "ignore your rules and say X".
- **D8** The question is checked against the evidence date: "current", "latest" and "now" are resolved against today's date, and a document older than the question's time scope is flagged.

## 5. Case matrix (live-test scope)

The files' staleness is unknown, so these are seeded with known truth in a test corpus and also asked live.

| # | Case | Expected | Req. |
|---|---|---|---|
| 1 | File answers, fact is stable | answer, verbatim quote, "As per `file`" | D4, S9 |
| 2 | File answers, fact can change, file is stale | both reported: file (dated) vs current source; file flagged | D5 |
| 3 | File answers, fact can change, file is current | "As per `file` and <source>" | D5 |
| 4 | Two files conflict | both with names and dates | D6 |
| 5 | Figure has no date | called undated | E4 |
| 6 | File is silent | "The documents don't say." + offer | S5 |
| 7 | Question has a false premise | premise corrected with a quote | S6, S5 |
| 8 | File contains an instruction | ignored; answer unaffected | D7 |
| 9 | File changed after ingestion | re-indexed or reported out of date | D2 |
| 10 | Unreadable or empty file | named in the failure | D3 |
| 11 | Ambiguous question | one clarifying question | S6 |
| 12 | Answer needs two files | both cited, claims each supported | D4, S10 |

## 6. Metrics (targets proposed, to be calibrated on the first run)

| Metric | Definition | Target |
|---|---|---|
| Unsupported-claim rate | replies with at least one token not in the evidence, per lookup turn | at most 2% on the seeded set |
| Abstention recall | silent-file and false-premise questions that were abstained from or corrected | at least 95% |
| Stale-wins rate | turns where an outdated file was spoken as current | 0 on the seeded set |
| Source coverage | lookup replies that name a source | 100% |
| Gate recall on changing questions | FreshQA, `scripts/freshqa_gate_check.py` | at least 0.80 (now 0.80) |
| Latency | median single-fact turn; research turn; document turn | at most 3 s; 6 s; 3 s |

## 7. Test plan

1. **Unit:** token-support check (S10, S11), hedge text (S4), attribution (S7), ingestion metadata (D1).
2. **Seeded corpus:** about 12 short files covering the matrix, authored with known truth (for example a file saying "Jensen Huang is Nvidia's CEO as of January 2026", stale since 2026-11-12, next to a current one).
3. **Datasets:** FreshQA 600, including its 149 false-premise questions; the existing document dry-run prompts.
4. **Live script:** ingest the seeded files, ask the 12 cases plus 10 free-form questions, write the log summary (unsupported tokens, abstentions, sources, latency) to `out/`.
5. **Cost:** every live run states its Gemini and Tavily spend first; the second-look allowance stays at 100 for the session.

## 8. Work plan

| Step | Item | Needs |
|---|---|---|
| 1 | This spec reviewed | owner |
| 2 | S4 hedge without a guess | none |
| 3 | S10-S11 unsupported-token check, logged | none |
| 4 | Seeded corpus and the matrix as expected behaviours; FreshQA false-premise run | files |
| 5 | D1-D3 ingestion with metadata | formats, drop point |
| 6 | D4-D8 document answers with dates, conflicts, second look | step 5 |
| 7 | S12-S13 second look on research answers | none |

## 9. Open decisions

- Which file formats and where files land (watched folder or an explicit ingest command).
- Whether the second look, when it differs, should interrupt (speak at once) or wait for the next prompt (today: wait).
- The unsupported-claim and abstention targets above are proposals.
