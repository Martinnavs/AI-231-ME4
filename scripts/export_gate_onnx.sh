#!/usr/bin/env bash
# export_gate_onnx.sh: build the changeable-question classifier (out/models/eg-e5/) for voice_agent.gate.
#   Downloads s-nlp/E5-EverGreen-Multilingual-Small (MIT, 471 MB), exports it to ONNX and quantizes it to int8
#   (118 MB, about 4 ms a question). Needs the network once, about 2 GB of scratch disk and a few minutes; the
#   torch/transformers it uses live in a throwaway venv (OUT/.export-venv, deleted at the end), not in the project.
#   Without the output the harness keeps using the regex. Never run this or the model on the Pi.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${GATE_OUT:-$REPO/out/models/eg-e5}
VENV=$OUT/.export-venv
mkdir -p "$OUT"
uv venv -q "$VENV"
trap 'rm -rf "$VENV"' EXIT
uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q --python "$VENV/bin/python" transformers safetensors sentencepiece onnx onnxscript onnxruntime
OUT=$OUT "$VENV/bin/python" - <<'PY'
import os
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

out = os.environ["OUT"]
name = "s-nlp/E5-EverGreen-Multilingual-Small"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()
assert model.config.id2label == {0: "Mutable", 1: "Evergreen"}, model.config.id2label  # gate.MUTABLE
enc = tok(["Who is the CEO of Nvidia?"], return_tensors="pt")


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


torch.onnx.export(
    Logits(model), (enc["input_ids"], enc["attention_mask"]), f"{out}/model.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "logits": {0: "b"}},
    opset_version=17, dynamo=False,
)
quantize_dynamic(f"{out}/model.onnx", f"{out}/model.int8.onnx", weight_type=QuantType.QInt8)
os.remove(f"{out}/model.onnx")
tok.backend_tokenizer.save(f"{out}/tokenizer.json")
print("wrote", out)
PY
ls -l "$OUT"
