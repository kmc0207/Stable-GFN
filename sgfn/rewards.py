"""Reward processing for S-GFN.

Pieces here are independent and composable:

* :func:`aggregate_lm_logreward` — turn a per-token LM log-prob tensor into a
  scalar per-sample log-reward. Implements ``"sum"``, ``"mean"`` and the
  paper's **Min-K Fluency Stabilizer** (``"bottom_k_mean"``).
* :class:`LMRewardClip` — hard clip below a threshold (replaces low-fluency
  rewards with a sentinel value).
* :class:`LinearScheduler` — linear scheduling of ``γ`` (LM-reward temperature)
  or ``β`` (classifier-reward scale), as used in the codebase.

Nothing here touches the network; pass tensors in, get tensors out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Aggregation = Literal["sum", "mean", "bottom_k_mean"]


# ---------------------------------------------------------------------------
# Aggregation: MKS lives here
# ---------------------------------------------------------------------------

def aggregate_lm_logreward(
    lm_logreward: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    aggregation: Aggregation = "sum",
    bottom_k: int = 5,
) -> torch.Tensor:
    """Aggregate per-token LM log-probabilities into a per-sample reward.

    Args:
      lm_logreward: ``[B, T]`` log-probability of each generated token.
      pad_mask: ``[B, T]`` True where the position is *padding* (i.e. ignored).
      aggregation:
        - ``"sum"``: ``Σ_t log π_ref(y_t | y_<t)`` (legacy, penalizes long y).
        - ``"mean"``: same but length-normalized.
        - ``"bottom_k_mean"``: **Min-K Fluency Stabilizer (MKS)**. Mean of the
          ``k`` lowest token-level log-probs per sequence; insensitive to
          length and to rare-but-licit tokens (e.g. proper nouns).
      bottom_k: ``k`` used by MKS. Paper uses 5–7.

    Returns:
      ``[B]`` tensor of aggregated LM log-rewards.
    """
    if aggregation == "mean":
        response_lengths = torch.sum((~pad_mask).float(), 1)
        return torch.sum(lm_logreward, 1) / torch.clamp(response_lengths, min=1.0)

    if aggregation == "bottom_k_mean":
        B = lm_logreward.size(0)
        out = torch.zeros(B, device=lm_logreward.device, dtype=lm_logreward.dtype)
        for i in range(B):
            valid = lm_logreward[i][~pad_mask[i]]
            if valid.numel() == 0:
                continue
            k = min(bottom_k, valid.numel())
            lowest, _ = torch.topk(valid, k, largest=False)
            out[i] = lowest.mean()
        return out

    # default "sum"
    return torch.sum(lm_logreward, 1)


# ---------------------------------------------------------------------------
# Reward clipping
# ---------------------------------------------------------------------------

@dataclass
class LMRewardClip:
    """Clip LM log-reward values below ``threshold`` to ``clipped_value``.

    Used to prevent the policy from being attracted to extremely-low-fluency
    out-of-distribution prompts (which the toxicity classifier sometimes
    scores spuriously well).

    Example:
      ``LMRewardClip(threshold=-10.0, clipped_value=-300.0)``
    """
    threshold: float = -200.0
    clipped_value: float = -300.0
    enabled: bool = True

    def __call__(self, lm_logreward: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return lm_logreward
        device = lm_logreward.device
        dtype = lm_logreward.dtype
        keep = torch.tensor(0.0, device=device, dtype=dtype)
        drop = torch.tensor(self.clipped_value, device=device, dtype=dtype)
        return torch.where(lm_logreward >= self.threshold, keep, drop)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

@dataclass
class LinearScheduler:
    """Linearly interpolate from ``start`` to ``end`` over ``horizon`` steps."""
    start: float
    end: float
    horizon: int

    def __call__(self, step: int) -> float:
        if self.horizon <= 0:
            return self.end
        progress = min(1.0, step / self.horizon)
        return self.start + (self.end - self.start) * progress


@dataclass
class NGPScheduler:
    """Schedule the NGP threshold σ.

    ``constant`` mode: returns ``base`` regardless of step.
    ``linear`` mode: interpolate from ``start`` to ``end`` over ``horizon``.
    ``step`` mode: returns ``start`` until ``horizon``, then ``end``.
    """
    base: float = 0.0
    start: float | None = None
    end: float | None = None
    horizon: int = 500
    kind: Literal["constant", "linear", "step"] = "constant"

    def __call__(self, step: int) -> float:
        if self.kind == "constant" or self.start is None or self.end is None:
            return self.base
        if self.kind == "step":
            return self.start if step < self.horizon else self.end
        # linear
        progress = min(1.0, step / max(1, self.horizon))
        return self.start + (self.end - self.start) * progress
