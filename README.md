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

## Verified end-to-end run

We verified the pipeline by running the full S-GFN training (600 steps) on
the configuration above, on 4 × RTX 4090:

| Stage                       | Wall time     |
|-----------------------------|---------------|
| Setup (vLLM + classifier)   | ~1 min        |
| Buffer init (30 SFT seeds)  | ~30 s         |
| Training (600 steps)        | ~52 min       |
| Eval (1024 attacks)         | ~4 min        |

Training reward trajectory (avg `c_log_reward = log p("unsafe")`):
`step 1: -3.46 → step 100: -2.66 → step 200: -0.21 → step 600: -0.08`,
i.e. the policy reaches ≈92% `p(unsafe)` on its own attacks.

Final eval (1024 attacks, single seed, `Meta-Llama-Guard-2-8B` as classifier):

| Metric | Verified run | Paper Table 1 (S-GFN, ±std) |
|--------|--------------|------------------------------|
| ASR    | **97.6%**    | 92.55 ± 2.87                  |
| UA     | **18**       | 134 ± 12.77                   |

**Caveat — diversity (UA) is sensitive to the toxicity classifier.** Our
verification run used `Meta-Llama-Guard-2-8B` (16 GB cached locally); the
paper used **`Llama-Guard-3-8B`** as in `configs/sgfn.yaml`. Guard-2 has a
markedly narrower toxicity surface: on this run the policy collapsed onto
variations of "(verb) strategies for growing cannabis indoors", which the
classifier rewards heavily — yielding very high ASR but few unique modes.
Reproducing the UA number from the paper requires switching to Guard-3
(set `model.classifier_model_id` in `configs/sgfn.yaml` and ensure that
model is downloaded locally).

### Full run with Llama-Guard-3-8B (paper default classifier)

Re-running the full pipeline end-to-end on **4 x RTX 4090** with the shipped
default config (`Llama-Guard-3-8B`), single seed 42:

| Stage                              | Wall time            |
|------------------------------------|----------------------|
| SFT warm-up (70 steps, full-FT)    | ~4 min               |
| S-GFN (600 steps, incl. setup)     | ~62 min (~6 s/step)  |
| Eval (1024 attacks x 5 responses) | ~4.5 min          |
| **Total**                          | **~70 min**          |

Reward trajectory (avg `c_log_reward = log p("unsafe")`, Guard-3):
`step 1: -4.84 -> 100: -4.49 -> 200: -3.91 -> 300: -0.12 -> 500: -0.03 -> 600: -0.12`,
i.e. the policy converges around step ~300 to ~89-97% `p(unsafe)`.

Final eval (1024 attacks, single seed, `Llama-Guard-3-8B`):

| Metric         | This run | Paper Table 1 (S-GFN, +/-std) |
|----------------|----------|-----------------------------------|
| ASR            | **97.3%**| 92.55 +/- 2.87                 |
| UA (thr 0.70)  | **40**   | 134 +/- 12.77                  |
| UA (thr 0.75)  | **78**   | -                           |

UA / ASR by training step (this run, 1024 attacks per checkpoint):

| Step | ASR    | UA @0.70 | UA @0.75 |
|------|--------|----------|----------|
| 100  | 0.9%   | 8        | 8        |
| 200  | 40.4%  | 35       | 36       |
| 300  | 96.4%  | 31       | 59       |
| 400  | 96.2%  | **91**   | **171**  |
| 500  | 99.0%  | 39       | 86       |
| 600  | 97.3%  | 40       | 78       |

**Diversity peaks mid-training, then collapses.** ASR saturates by step ~300
and stays >96%, but UA peaks at step 400 (91 @0.70 / 171 @0.75 -- near the
paper's 134) and then drops by half as the policy collapses onto a few
high-reward modes. The final (600) checkpoint is therefore NOT the most
diverse: select by best UA on a held-out eval (or early-stop) rather than
always taking `latest`. UA is also sensitive to the clustering threshold and
to single-seed variance.

---

## Reproducing Table 1

```bash
# Attacker SFT (run once, ~30 min on 1 × 4090)
bash scripts/train_sft.sh 42

# S-GFN (Ours) — paper default
bash scripts/train_sgfn.sh 42

# GFN-TB baseline
bash scripts/train_gfn_tb.sh 42

# For other seeds:
bash scripts/train_sgfn.sh 0
bash scripts/train_sgfn.sh 1
```

Each run saves a checkpoint and a buffer snapshot under
`save/<exp_name>/<step>/`. To evaluate ASR / number of unique attacks, generate
1024 attacks per run and compute the metrics with the eval scripts of your
choice (we used greedy clustering at threshold 0.75 with
`sentence-transformers/all-MiniLM-L6-v2`, following Yun et al. 2025).

---

## Notes & caveats

- The original codebase contained several exploratory features (Renyi-α loss,
  diversity rewards via RND, length-bonus shaping, "active attacks"
  alternation, several debug paths) that did not make it into the paper. They
  are removed in this release to keep the surface small.
- The Llama-Guard classifier is loaded as a CausalLM and we read
  ``log p("unsafe")`` from the next-token distribution (Section 5.1).
- vLLM is initialized **before** wandb and the attacker model on purpose; do
  not reorder.
- The replay buffer is seeded from the SFT toxic-instruction set on the first
  run. To reuse a pre-built buffer set ``buffer.buffer_path`` in the YAML
  config.

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

## License

MIT. See `LICENSE`.
