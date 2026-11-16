"""Does this question's answer change over time? A small classifier, with the regex as fallback.

`s-nlp/E5-EverGreen-Multilingual-Small` (Pletenev et al. 2025, MIT; arXiv 2505.21115) labels a
question Mutable or Evergreen from its text. Exported to ONNX and quantized to int8 by
`scripts/export_gate_onnx.sh` (118 MB, about 4 ms a question on one CPU thread), it flags 0.80 of
FreshQA's changing questions at 0.18 false alarms; the regex it replaces flags 0.55 at 0.21
(`scripts/freshqa_gate_check.py`). Needs `onnxruntime` and `tokenizers`; with either or the
model files missing, `mutable` returns None and the caller uses the regex. It must never run on
the Pi (no model there but Silero): the Pi has no `out/models/eg-e5`, so it falls back by itself.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_DIR = Path(
    os.environ.get("CHANGEABLE_GATE_DIR")
    or Path(__file__).resolve().parents[2] / "out/models/eg-e5"
)
THRESHOLD = (
    0.5  # P(mutable) at or above this: changeable (0.1-0.7 barely moves recall or false alarms)
)
MAX_TOKENS = 64
MUTABLE = 0  # the model's label 0 is Mutable, 1 is Evergreen


class MutableGate:
    def __init__(self, model_dir: Path) -> None:
        import numpy as np  # noqa: PLC0415 (only with the model)
        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # a question is short: more threads only add overhead
        self.session = ort.InferenceSession(
            str(model_dir / "model.int8.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)

    def p_mutable(self, question: str) -> float:
        np = self.np
        encoded = self.tokenizer.encode(question)
        logits = self.session.run(
            None,
            {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            },
        )[0][0]
        weights = np.exp(logits - logits.max())
        return float((weights / weights.sum())[MUTABLE])


_loaded: list[MutableGate | None] = []  # [gate or None], filled once


def load(model_dir: Path | None = None) -> MutableGate | None:
    """The gate, or None when its files or libraries are missing (said once in the log)."""
    if _loaded:
        return _loaded[0]
    path = model_dir or MODEL_DIR
    try:
        gate: MutableGate | None = MutableGate(path)
    except Exception as exc:  # onnxruntime raises its own types (NoSuchFile, Fail, InvalidProtobuf)
        log.info("changeable-question gate not loaded (%s): using the regex", exc)
        gate = None
    _loaded.append(gate)
    return gate


def reset() -> None:
    """Forget the loaded gate (tests)."""
    _loaded.clear()


def mutable(question: str) -> bool | None:
    """True or False from the model, None when there is none (or it failed on this question)."""
    gate = load()
    if gate is None:
        return None
    try:
        return gate.p_mutable(question) >= THRESHOLD
    except Exception:  # a failure on one question is the regex's to answer
        return None
