#!/usr/bin/env bash
# GFN-TB baseline (Trajectory Balance with learnable log Z).
# Same setup as S-GFN but with the original TB loss; used for Table 1.
#
# Usage:
#   bash scripts/train_gfn_tb.sh [SEED]
set -euo pipefail

SEED="${1:-42}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
sgfn train \
    --config configs/gfn_tb.yaml \
    --set seed=${SEED} \
          io.exp_name="\"gfn-tb-qwen/seed${SEED}\""
