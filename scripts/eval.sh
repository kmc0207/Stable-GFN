#!/usr/bin/env bash
# Evaluate a trained attacker: ASR + Unique-Attacks against a victim.
#
# Usage:
#   bash scripts/eval.sh CKPT_DIR [OUTPUT_DIR] [NUM_SAMPLES]
#
# Example:
#   bash scripts/eval.sh save/sgfn-qwen/latest results/sgfn-qwen 1024
set -euo pipefail

CKPT="${1:?expected path to attacker checkpoint}"
OUT="${2:-results/eval}"
N="${3:-1024}"

sgfn eval \
    --config configs/eval.yaml \
    --ckpt "${CKPT}" \
    --output-dir "${OUT}" \
    --num-samples "${N}"
