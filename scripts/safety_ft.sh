#!/usr/bin/env bash
# Safety fine-tune the victim model on a JSON of (instruction, refusal) pairs.
# The resulting LoRA adapter becomes the "defended" victim used in the
# cross-attack evaluation (Section 5.2 of the paper).
#
# Usage:
#   bash scripts/safety_ft.sh [SAFETY_DATASET_JSON] [EXP_NAME]
set -euo pipefail

DATASET="${1:-prompts/safety_datasets/sgfn_refusals.json}"
EXP_NAME="${2:-safety-ft-sgfn}"

sgfn train \
    --config configs/safety.yaml \
    --set io.prompt_file="\"${DATASET}\"" \
          io.exp_name="\"${EXP_NAME}\""
