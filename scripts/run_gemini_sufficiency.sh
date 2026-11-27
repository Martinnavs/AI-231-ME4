#!/usr/bin/env bash
# Paid: ~50 Gemini Flash-Lite verdict calls per run, dev split. Run it yourself: ! bash scripts/run_gemini_sufficiency.sh
set -euo pipefail
cd "$(dirname "$0")/../pipeline"
mkdir -p ../out/sufficiency_eval
common=(--set both --split dev --verdict-backend gemini --today 2026-11-27)
uv run python ../scripts/sufficiency_eval.py "${common[@]}" --gold-answer --out ../out/sufficiency_eval/gemini-dev-gold.json
uv run python ../scripts/sufficiency_eval.py "${common[@]}" --out ../out/sufficiency_eval/gemini-dev-lane.json  # needs the Qwen relay on 127.0.0.1:18080
