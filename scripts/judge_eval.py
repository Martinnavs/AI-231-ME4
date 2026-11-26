"""Grade answers with the local Qwen relay as a "Decisions"-style judge (ticket 246).

Each metric is one prompt with single-letter options (A/B/C/D) so the first output token carries
the choice. The choice and the per-option probabilities come from the relay's top_logprobs
(softmax over the option letters only); the probability of the chosen letter is the confidence.
The judge sees ONLY the material in the prompt and never outside knowledge.

    .venv/bin/python scripts/judge_eval.py turns --log pipeline/out/agent_server/<run>.jsonl \
        [--metrics directness,faithfulness] [--explain] [--out FILE]
    .venv/bin/python scripts/judge_eval.py rows --in rows.json [--explain]

rows.json is a JSON list of {id?, question, answer, evidence?, gold?, key_points?, reference?,
excerpts?} (evidence/gold may be a string or a list).

Metrics and the inputs they need (absent inputs -> no row for that item, not a silent skip of a
call; a call that still fails after one retry stops the run loudly):
  directness    question, answer            A answers / B describes_only / C not_found
  faithfulness  question, answer, evidence  A supported / B partial / C unsupported / D contradicted
  correctness   question, answer, gold      A supported / B partial / C missing / D contradicted
  completeness  answer, one key point       A covered / B partial / C not_covered
  recall        reference, excerpts         A found_in_one_excerpt / B found_across_several /
                                            C partly_found / D not_found
  precision     one excerpt, reference      A key_information / B side_detail / C irrelevant

From an agent-server log (turns mode) the evidence is the calls[].result_text of the turn
(search_docs JSON is flattened to its passage texts); log rows may also carry gold, key_points,
reference, excerpts or passages directly.

Output: one JSON line per judged item {turn, question, metric, choice, conf, probs, why} plus a
summary table (counts per metric/choice and the lowest-confidence rows). `--explain` re-asks every
call whose confidence is below 0.8 or whose choice is not the top-1 letter, with thinking on
(max_tokens 400), for one short reason stored in "why".

The relay is OpenAI-compatible (http://127.0.0.1:18080/v1 by default, model Qwen/Qwen3.8-27B).
The key comes from ~/.secrets/api_key.txt (JUDGE_API_KEY overrides it, JUDGE_SECRETS_DIR points
at another folder) and travels only in the Authorization header: it never appears in output,
logs or errors. Calls run one at a time: the relay is shared with coding runs.

calibrate(rows, known, metric=None) compares a finished run against known expected choices
({turn: letter}) and reports agreement; intended for turns already known good or bad.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
KEY_FILE = "api_key.txt"
EXPLAIN_BELOW = 0.8
RETRY_PAUSE_S = 1.0
MAX_ITEM_CHARS = 4000
EVIDENCE_CAP = 8

SYSTEM = (
    "You are a strict judge of a spoken answer. Use ONLY the material given below; never use "
    "outside knowledge. Pick the option that fits best and answer with exactly one single "
    "letter, the letter of the option, and nothing else."
)

EXPLAIN_SYSTEM = (
    "You are a strict judge of a spoken answer. Use ONLY the material given below; never use "
    "outside knowledge. Answer with one short sentence and nothing else."
)

# metric -> {letter: (label, option description)}
METRICS: dict[str, dict[str, tuple[str, str]]] = {
    "directness": {
        "A": (
            "answers",
            "the first sentence of the answer states the specific thing the question asks for",
        ),
        "B": (
            "describes_only",
            "the answer talks about the topic but never states the specific thing asked",
        ),
        "C": (
            "not_found",
            "the answer says it cannot find the information, or does not address the question",
        ),
    },
    "faithfulness": {
        "A": ("supported", "every fact the answer states is contained in the evidence"),
        "B": ("partial", "some of what the answer states is in the evidence and some is not"),
        "C": ("unsupported", "the answer states things the evidence does not contain"),
        "D": ("contradicted", "the answer states something the evidence says is not the case"),
    },
    "correctness": {
        "A": ("supported", "the answer contains the expected fact or facts"),
        "B": ("partial", "the answer contains only some of the expected facts"),
        "C": ("missing", "the answer contains none of the expected facts"),
        "D": ("contradicted", "the answer states one of the expected facts wrongly"),
    },
    "completeness": {
        "A": ("covered", "the answer fully covers the key point"),
        "B": ("partial", "the answer covers only part of the key point"),
        "C": ("not_covered", "the answer does not cover the key point"),
    },
    "recall": {
        "A": (
            "found_in_one_excerpt",
            "the reference information is contained in a single excerpt",
        ),
        "B": (
            "found_across_several",
            "the reference information is spread over two or more excerpts",
        ),
        "C": ("partly_found", "only part of the reference information appears in the excerpts"),
        "D": ("not_found", "none of the reference information appears in the excerpts"),
    },
    "precision": {
        "A": ("key_information", "the excerpt contains important information from the reference"),
        "B": ("side_detail", "the excerpt is about the same subject but only a side detail"),
        "C": ("irrelevant", "the excerpt does not concern the reference"),
    },
}

Job = dict[str, Any]


class Transport(Protocol):
    """Anything that can send one judge request and return the parsed relay response."""

    def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class JudgeError(RuntimeError):
    """A loud failure: the run stops instead of skipping the item."""


def label_for(metric: str, choice: str) -> str:
    return METRICS[metric].get(choice, ("?", "?"))[0]


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                out.append(str(text) if isinstance(text, str) and text.strip() else "")
            else:
                out.append(str(item))
        return [t for t in (x.strip() for x in out) if t]
    return [str(value)] if str(value).strip() else []


def _clip(text: str, cap: int = MAX_ITEM_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= cap else text[:cap] + " [...]"


def _clean_result(tool: str, raw: str) -> str:
    """search_docs results are JSON lists of passages; flatten them to plain text for the judge."""
    if tool != "search_docs" or not raw.lstrip().startswith(("[", "{")):
        return raw
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    items = data if isinstance(data, list) else [data]
    texts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            for key in ("text", "page_content", "passage"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
    return "\n".join(texts) if texts else raw


def _evidence_from_calls(row: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for call in row.get("calls") or []:
        if not isinstance(call, dict) or call.get("status") == "cancelled":
            continue
        raw = call.get("result_text")
        if isinstance(raw, str) and raw.strip():
            texts.append(_clip(_clean_result(str(call.get("tool") or ""), raw)))
        for source in call.get("sources") or []:
            if isinstance(source, str) and source.strip():
                texts.append(source.strip())
    return texts[:EVIDENCE_CAP]


def jobs_from_log_row(row: Mapping[str, Any], index: int) -> Job | None:
    """One agent-server log row (or one rows-mode entry) into a judge job, or None if there is
    no question and no answer to judge."""
    question = str(row.get("question") or "").strip()
    answer = str(row.get("text") or row.get("answer") or "").strip()
    if not question and not answer:
        return None
    evidence = _as_text_list(row.get("evidence")) or _evidence_from_calls(row)
    reference = _clip(str(row.get("reference") or "").strip())
    excerpts = [_clip(e) for e in _as_text_list(row.get("excerpts"))]
    if not excerpts:
        excerpts = [_clip(e) for e in _as_text_list(row.get("passages"))]
    return {
        "turn": row.get("turn", index),
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "gold": [_clip(g) for g in _as_text_list(row.get("gold"))],
        "key_points": [_clip(k) for k in _as_text_list(row.get("key_points"))],
        "reference": reference,
        "excerpts": excerpts,
    }


def load_log_rows(path: Path) -> tuple[list[Job], int]:
    """(judge jobs, number of skipped turns without a question or answer text)."""
    jobs: list[Job] = []
    skipped = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        job = jobs_from_log_row(json.loads(line), line_no)
        if job is None:
            skipped += 1
        else:
            jobs.append(job)
    return jobs, skipped


def load_rows_input(path: Path) -> list[Job]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise JudgeError(f"{path}: expected a JSON list of rows")
    jobs: list[Job] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise JudgeError(f"{path}: row {i} is not an object")
        job = jobs_from_log_row(item, i)
        if job is None:
            raise JudgeError(f"{path}: row {i} has no question or answer")
        if item.get("id") is not None:
            job["turn"] = item["id"]
        jobs.append(job)
    return jobs


def build_user_prompt(metric: str, job: Job, point: str | None = None) -> str:
    q, a = job["question"], job["answer"]
    if metric == "directness":
        body = f"QUESTION:\n{q}\n\nANSWER:\n{a}"
    elif metric == "faithfulness":
        body = f"QUESTION:\n{q}\n\nANSWER:\n{a}\n\nEVIDENCE:\n{chr(10).join(job['evidence'])}"
    elif metric == "correctness":
        body = f"QUESTION:\n{q}\n\nANSWER:\n{a}\n\nEXPECTED FACT(S):\n{chr(10).join(job['gold'])}"
    elif metric == "completeness":
        body = f"ANSWER:\n{a}\n\nKEY POINT:\n{point or ''}"
    elif metric == "recall":
        excerpts = _numbered(job["excerpts"])
        body = f"REFERENCE PASSAGE:\n{job['reference']}\n\nRETRIEVED EXCERPTS:\n{excerpts}"
    elif metric == "precision":
        body = f"EXCERPT:\n{point or ''}\n\nREFERENCE PASSAGE(S):\n{job['reference']}"
    else:
        raise JudgeError(f"unknown metric {metric!r}")
    options = "\n".join(
        f"{letter}: {label} - {description}"
        for letter, (label, description) in METRICS[metric].items()
    )
    return f"{body}\n\nOptions:\n{options}\n\nAnswer with one letter only."


def _numbered(items: Sequence[str]) -> str:
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(items, start=1))


def judge_payload(
    model: str, user_prompt: str, *, thinking: bool = False, max_tokens: int = 4
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXPLAIN_SYSTEM if thinking else SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if not thinking:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    return payload


def _token_letter(token: str, allowed: str) -> str | None:
    """The option letter the token commits to: the token must be a single letter (surrounding
    whitespace and punctuation ignored). Word tokens like "am" or "About" are never a choice."""
    core = token.strip().strip(".,!?;:'\"()[]{}-").strip()
    if len(core) == 1 and core.upper() in allowed:
        return core.upper()
    return None


def parse_letter(
    data: Mapping[str, Any], options: Mapping[str, tuple[str, str]]
) -> tuple[str, dict[str, float]]:
    """(choice, per-option probabilities) from the first token's top_logprobs.

    The probabilities are the softmax restricted to the option letters: each option letter's
    logprob is exponentiated and the values are renormalized over the option letters present in
    the top-5. Letters absent from the top-5 get 0 (their probability was below the cut)."""
    try:
        choice_data = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError("the relay response has no choices") from exc
    allowed = "".join(options)
    raw: dict[str, float] = {}
    token_text = ""
    content = (choice_data.get("logprobs") or {}).get("content") or []
    if content:
        first = content[0]
        token_text = str(first.get("token") or "")
        for item in first.get("top_logprobs") or []:
            letter = _token_letter(str(item.get("token") or ""), allowed)
            if letter and "logprob" in item:
                raw[letter] = math.exp(float(item["logprob"]))
    if not token_text:
        token_text = str((choice_data.get("message") or {}).get("content") or "")
    choice = _token_letter(token_text, allowed)
    if choice is None:
        if raw:
            choice = _argmax(raw)
        else:
            raise JudgeError(
                f"the judge answered without an option letter (first token {token_text[:20]!r})"
            )
    total = sum(raw.values())
    probs = {letter: (raw.get(letter, 0.0) / total if total > 0 else 0.0) for letter in options}
    return choice, probs


def _argmax(probs: Mapping[str, float]) -> str:
    return max(probs, key=lambda letter: (probs[letter], -ord(letter)))


class Relay:
    """The OpenAI-compatible relay: one call at a time, one retry on HTTP or connection errors,
    a loud JudgeError otherwise. The key lives only in the request headers."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def request(self, payload: Mapping[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )

    def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(self.request(payload), timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if attempt == 2:
                    raise JudgeError(f"the relay answered HTTP {exc.code} (retried once)") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise JudgeError(f"the relay is unreachable ({type(exc).__name__})") from None
            else:
                if not isinstance(data, dict):
                    raise JudgeError("the relay answered with a non-JSON object")
                return data
            time.sleep(RETRY_PAUSE_S)
        raise AssertionError("unreachable: the last attempt always returns or raises")


def read_key(secrets_dir: str | Path | None = None) -> str:
    env_key = os.environ.get("JUDGE_API_KEY", "").strip()
    if env_key:
        return env_key
    directory = Path(secrets_dir or os.environ.get("JUDGE_SECRETS_DIR") or Path.home() / ".secrets")
    path = directory / KEY_FILE
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        raise JudgeError(f"cannot read the relay key from {path}") from None
    if not key:
        raise JudgeError(f"the relay key file is empty: {path}")
    return key


def applicable(metric: str, job: Job, point: str | None = None) -> bool:
    q, a = job["question"], job["answer"]
    if metric == "directness":
        return bool(q and a)
    if metric == "faithfulness":
        return bool(q and a and job["evidence"])
    if metric == "correctness":
        return bool(q and a and job["gold"])
    if metric == "completeness":
        return bool(a and point)
    if metric in ("recall", "precision"):
        return bool(job["reference"] and (job["excerpts"] if metric == "recall" else point))
    return False


def _points_for(metric: str, job: Job) -> list[str | None]:
    """One call per point: one completeness call per key point, one precision call per excerpt."""
    if metric == "completeness":
        return (job["key_points"] or job["gold"]) or [None]
    if metric == "precision":
        return job["excerpts"] or [None]
    return [None]


def run_jobs(
    jobs: Sequence[Job],
    metrics: Sequence[str],
    transport: Transport,
    model: str,
    explain: bool = False,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Judge every (job, metric, point); strictly one call at a time, no concurrency."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        for metric in metrics:
            for point in _points_for(metric, job):
                if not applicable(metric, job, point):
                    continue
                prompt = build_user_prompt(metric, job, point)
                choice, probs = parse_letter(
                    transport.post(judge_payload(model, prompt)), METRICS[metric]
                )
                conf = probs.get(choice, 0.0)
                row = {
                    "turn": job["turn"],
                    "question": job["question"],
                    "metric": metric,
                    "choice": choice,
                    "conf": round(conf, 6),
                    "probs": {letter: round(value, 6) for letter, value in probs.items()},
                    "why": None,
                }
                if explain and (conf < EXPLAIN_BELOW or choice != _argmax(probs)):
                    row["why"] = _explain(transport, model, prompt, metric, choice)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return rows


def _explain(transport: Transport, model: str, prompt: str, metric: str, choice: str) -> str:
    text = (
        f"{prompt}\n\nYour choice was {choice} ({label_for(metric, choice)}). "
        "In one short sentence, say only the reason."
    )
    data = transport.post(judge_payload(model, text, thinking=True, max_tokens=400))
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "no reason given"
    why = str(message.get("content") or message.get("reasoning_content") or "").strip()
    why = re.sub(r"^thinking\s*[:\-]?\s*", "", why, flags=re.IGNORECASE)
    return why or "no reason given"


def summarize(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "summary: no rows"
    by_metric: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(str(row["metric"]), []).append(row)
    lines = ["summary (counts per metric/choice):"]
    for metric in sorted(by_metric):
        sub = by_metric[metric]
        counts = Counter(str(r["choice"]) for r in sub)
        cells = " ".join(f"{counts.get(letter, 0):>4}" for letter in "ABCD")
        confs = [float(r["conf"]) for r in sub]
        lines.append(
            f"{metric:<14} {cells}  n={len(sub):<4} avg_conf={sum(confs) / len(confs):.2f}"
        )
    lines.append("lowest confidence:")
    for row in sorted(rows, key=lambda r: float(r["conf"]))[:5]:
        question = str(row["question"])[:60]
        lines.append(
            f"  turn {row['turn']}  {row['metric']:<13} {row['choice']}  "
            f"conf={float(row['conf']):.2f}  {question}"
        )
    return "\n".join(lines)


def calibrate(
    rows: Sequence[Mapping[str, Any]],
    known: Mapping[Any, str],
    metric: str | None = None,
) -> dict[str, Any]:
    """Agreement of a finished run against known expected choices.

    rows: judge output rows ({turn, metric, choice, ...}); known: {turn: expected letter}.
    When several rows share a turn (one per metric/point), `metric` picks which to compare."""
    picked: dict[Any, str] = {}
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        picked.setdefault(row["turn"], str(row["choice"]))
    matched = agreed = 0
    missing: list[Any] = []
    disagreements: list[dict[str, Any]] = []
    for turn, expected in known.items():
        got = picked.get(turn)
        if got is None:
            missing.append(turn)
            continue
        matched += 1
        if got == str(expected):
            agreed += 1
        else:
            disagreements.append({"turn": turn, "expected": str(expected), "got": got})
    return {
        "n": len(known),
        "matched": matched,
        "missing": missing,
        "disagreements": disagreements,
        "agreement": (agreed / matched) if matched else None,
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Qwen logprob judge (ticket 246)")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, help_text in (
        ("turns", "judge an agent-server JSONL log (question, text, calls[].result_text)"),
        ("rows", "judge a JSON list of {question, answer, evidence?, gold?, ...}"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name == "turns":
            p.add_argument("--log", type=Path, required=True, help="agent-server JSONL log")
        else:
            p.add_argument("--in", dest="in_", type=Path, required=True, help="JSON list file")
        p.add_argument(
            "--metrics",
            default=",".join(METRICS),
            help=f"comma list from {', '.join(METRICS)} (default: all)",
        )
        p.add_argument(
            "--explain",
            action="store_true",
            help="re-ask (thinking on) calls with conf < 0.8 or a non-top-1-letter choice",
        )
        p.add_argument(
            "--out",
            type=Path,
            default=None,
            help="output JSONL (default out/judge_eval/<stamp>.jsonl; use - for stdout)",
        )
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--model", default=DEFAULT_MODEL)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, transport: Any = None, model: str | None = None) -> int:
    args = parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        print(
            f"unknown metrics: {', '.join(unknown)} (have: {', '.join(METRICS)})", file=sys.stderr
        )
        return 2
    if args.mode == "turns":
        jobs, skipped = load_log_rows(args.log)
        if skipped:
            print(f"skipped {skipped} turns with no question or answer text")
    else:
        jobs = load_rows_input(args.in_)

    if transport is None:  # real relay: key from the secrets dir, never printed
        relay = Relay(
            os.environ.get("JUDGE_BASE_URL") or args.base_url,
            read_key(),
            os.environ.get("JUDGE_MODEL") or args.model,
        )
        transport = relay
        model = relay.model
    if model is None:
        model = args.model

    rows = run_jobs(jobs, metrics, transport, model, explain=args.explain)

    out_lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    out_path = args.out
    if out_path is None:
        out_path = Path("out/judge_eval") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    if out_path == Path("-"):
        sys.stdout.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {out_path}")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
