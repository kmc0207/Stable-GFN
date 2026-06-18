#!/usr/bin/env bash
# Train S-GFN (paper default: CTB + NGP + MKS).
# Assumes an SFT checkpoint is already produced via scripts/train_sft.sh.
#
# Usage:
#   bash scripts/train_sgfn.sh [SEED]
set -euo pipefail

SEED="${1:-42}"

# 4 GPUs: victim on GPU 0, classifier on GPU 1+2, attacker on GPU 3.
# Adjust via --set gpu.attacker_devices=[...] etc. or by exporting
# CUDA_VISIBLE_DEVICES if you only want a subset visible.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
sgfn train \
    --config configs/sgfn.yaml \
    --set seed=${SEED} \
          io.exp_name="\"sgfn-qwen/seed${SEED}\""
