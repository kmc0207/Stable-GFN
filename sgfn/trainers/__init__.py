"""Trainers for the three stages of S-GFN.

* :class:`SFTTrainer`    — initial SFT warm-up of the attacker policy.
* :class:`GFNTrainer`    — main S-GFN trainer with CTB / NGP / MKS.
* :class:`SafetyTrainer` — safety fine-tuning of the victim model
  (used for the cross-attack evaluation in the paper).
"""

from .sft import SFTTrainer
from .safety import SafetyTrainer
from .gfn import GFNTrainer

__all__ = ["SFTTrainer", "SafetyTrainer", "GFNTrainer"]
