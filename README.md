# Stable-GFlowNet (S-GFN)

Diverse and robust LLM red-teaming via **Contrastive Trajectory Balance**.

This repository contains the reference implementation of:

- **CTB** — Contrastive Trajectory Balance (pairwise GFlowNet loss, no `log Z`).
- **NGP** — Noisy Gradient Pruning (mask pairs with `|Δ log R| < σ`).
- **MKS** — Min-K Fluency Stabilizer (token-level fluency penalty for the
  reference-model reward).

The pipeline is:

```
SFT warm-up (attacker)  →  S-GFN red-teaming  →  Safety fine-tuning (victim)
```

---

## Installation

```bash
git clone https://github.com/<org>/sgfn.git
cd sgfn
python -m venv .venv && source .venv/bin/activate

# Install vLLM FIRST, then everything else.
pip install vllm                # pins a compatible torch automatically
pip install -r requirements.txt # the rest (torch already satisfied by vLLM)
pip install -e .                # exposes the `sgfn` console script
```

> **Install order matters.** vLLM is the component with the strictest
> dependency constraints (it pins an exact `torch` build). Install `vllm`
> **first** and let it pull in the torch version it wants, then install the
> remaining requirements on top — this avoids pip resolving torch to a
> version that vLLM later rejects. The `torch` upper bound in
> `requirements.txt` mainly documents the version the paper used and is not
> load-bearing, so whatever torch vLLM selects is fine.

> **GPU.** A single 24 GB GPU is enough for the SFT and safety-FT stages.
> S-GFN red-teaming as configured in the paper uses 4 × RTX 4090 (one each
> for the victim, attacker, and two for the Llama-Guard classifier). The
> placement is fully configurable via `configs/sgfn.yaml > gpu`.

> **Models.** Default victim is `Qwen/Qwen2.5-1.5B-Instruct`; default
> classifier is `meta-llama/Llama-Guard-3-8B`. Both require accepting the
> license on Hugging Face and `huggingface-cli login`.

---

## Quickstart

```bash
# 1. SFT the attacker on the harmful-instruction set
bash scripts/train_sft.sh

# 2. Train S-GFN (paper default)
bash scripts/train_sgfn.sh

# 3. Evaluate ASR + Unique Attacks
bash scripts/eval.sh save/sgfn-qwen/latest results/sgfn-qwen 1024

# 4. (optional) cross-attack defense: fine-tune the victim on S-GFN-found refusals
bash scripts/safety_ft.sh
```

Every script forwards to `sgfn train --config <yaml> --set k=v ...`. You can
inspect the resolved config without running anything:

```bash
sgfn show-config --config configs/sgfn.yaml --set ngp.sigma=0.5
```

---

## Configs

| File                  | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `configs/sft.yaml`    | SFT warm-up of the attacker (Qwen2.5-1.5B).                  |
| `configs/sgfn.yaml`   | **S-GFN paper default**: CTB + NGP(σ=0.1) + MKS(k=7).         |
| `configs/gfn_tb.yaml` | GFN-TB baseline (vanilla Trajectory Balance, no NGP, no MKS). |
| `configs/safety.yaml` | LoRA safety fine-tuning of a victim model.                   |
| `configs/eval.yaml`   | ASR / UA evaluation against a victim model.                  |

Common overrides:

```bash
# Turn NGP off (CTB only)
sgfn train --config configs/sgfn.yaml --set ngp.sigma=0.0

# Switch to mean-baseline CTB
sgfn train --config configs/sgfn.yaml --set loss.baseline_mode='"mean_baseline"'

# Different MKS k
sgfn train --config configs/sgfn.yaml --set reward.lm_bottom_k=5

# Smaller paired-sampling on-policy ratio (4 on-policy / 8 off-policy in B=12)
sgfn train --config configs/sgfn.yaml --set buffer.paired_on_policy_ratio=0.33

# Map to different GPUs
sgfn train --config configs/sgfn.yaml \
    --set gpu.attacker_devices='[0]' gpu.victim_device=1 gpu.classifier_devices='[2,3]'
```

---

## Evaluation

```bash
# Evaluate a checkpoint with the default eval config:
bash scripts/eval.sh save/sgfn-qwen/latest results/sgfn-qwen 1024

# Equivalent direct CLI call (full control over batch size / sampling):
sgfn eval \
    --config configs/eval.yaml \
    --ckpt   save/sgfn-qwen/latest \
    --output-dir results/sgfn-qwen \
    --num-samples 1024 \
    --attacker-batch-size 32 \
    --victim-batch-size 16 \
    --num-responses-per-attack 5

# Eval against a defended victim (a safety-fine-tuned LoRA on the victim):
sgfn eval \
    --config configs/eval.yaml \
    --ckpt   save/sgfn-qwen/latest \
    --output-dir results/sgfn-vs-safety-ft \
    --set model.victim_name='"save/safety-ft-sgfn/latest"'
```

Outputs (under `--output-dir`):

* `attacks.json` — raw attack prompts (1024 by default).
* `scored.json`  — per-attack victim responses + avg/std toxicity.
* `metrics.json` — summary: **ASR**, **Unique Attacks**, diversity.

The eval pipeline uses **plain `transformers.generate`** for the victim (no
vLLM needed), which makes it usable on a single GPU; only the classifier
remains heavy (Llama-Guard-3-8B → ~16 GB).

---

## Package layout

```
sgfn/
├── losses.py        TB and Contrastive Trajectory Balance (with NGP)
├── rewards.py       Min-K Fluency Stabilizer + reward clip / schedulers
├── buffer.py        ReplayBuffer (edit-distance) and CosineReplayBuffer
├── classifiers.py   Llama-Guard, HarmAug, Roberta, StringMatch
├── generation.py    Autoregressive sampling with per-token log-probs + log Z head
├── data.py          SFTDataset, SafetyDataset, RedTeamDataset
├── config.py        Typed, YAML-backed SGFNConfig
├── eval.py          collect_attacks → score_attacks → compute_metrics (ASR / UA)
├── utils.py         seed / LoRA / iterator / decay-param helpers
├── cli.py           `sgfn` console script + `python -m sgfn`
├── __main__.py      Entry for `python -m sgfn`
└── trainers/
    ├── sft.py       SFTTrainer
    ├── gfn.py       GFNTrainer (the main S-GFN trainer)
    └── safety.py    SafetyTrainer
```

Each module is small enough to read end-to-end and the loss/reward functions
are pure (no `args` reference). If you want to drop CTB or MKS into another
GFlowNet codebase, copy `sgfn/losses.py` and `sgfn/rewards.py` directly.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{sgfn2026,
  title     = {Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming
               via Contrastive Trajectory Balance},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
