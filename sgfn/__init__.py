"""Stable-GFlowNet (S-GFN) for diverse LLM red-teaming.

Public modules:
  - sgfn.losses     : TB and Contrastive Trajectory Balance (CTB) losses
  - sgfn.rewards    : Min-K Fluency Stabilizer (MKS), reward clipping/schedulers
  - sgfn.buffer     : replay buffer with edit-distance / cosine deduplication
  - sgfn.classifiers: Llama-Guard, HarmAug, Roberta, string-match classifiers
  - sgfn.generation : on-policy sampling with per-token log-probabilities
  - sgfn.data       : datasets used for SFT / safety FT / red-teaming
  - sgfn.config     : YAML-backed dataclass configs
  - sgfn.trainers   : SFTTrainer, GFNTrainer, SafetyTrainer
  - sgfn.cli        : `sgfn` / `python -m sgfn` command-line interface
"""

__version__ = "0.1.0"
