#!/usr/bin/env bash
# Initial SFT warm-up of the attacker policy.
# Produces a checkpoint at save/qwen2.5-1.5b-sft/latest, which is the starting
# point for `train_sgfn.sh`.
#
# Usage:
#   bash scripts/train_sft.sh [SEED]
set -euo pipefail

SEED="${1:-42}"

sgfn train \
    --config configs/sft.yaml \
    --set seed=${SEED}
