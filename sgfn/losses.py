"""GFlowNet losses for LLM red-teaming.

Two losses are exposed:

* :func:`tb_loss` — original Trajectory Balance with a learnable ``log Z``.
* :func:`ctb_loss` — Contrastive Trajectory Balance (this paper). Three
  baseline modes are supported; the default ``"all_pairs"`` matches the
  formulation in the paper. NGP (Noisy Gradient Pruning) is integrated as a
  ``min_reward_diff`` mask.

Both functions are pure: they take tensors in and return tensors out, with no
references to a trainer-level ``args`` object. This makes them easy to drop
into other GFlowNet codebases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


BaselineMode = Literal["all_pairs", "mean_baseline", "median_baseline"]


# ---------------------------------------------------------------------------
# Standard Trajectory Balance (baseline)
# ---------------------------------------------------------------------------

def tb_loss(
    log_z: torch.Tensor,
    sum_logpf: torch.Tensor,
    log_reward: torch.Tensor,
    log_z_bias: float = 0.0,
) -> torch.Tensor:
    """Original TB loss: ``(log Z + log PF(y) - log R(y))^2``.

    Shapes are all ``[B]``.
    """
    if log_z_bias != 0.0:
        log_z = log_z + log_z_bias
    delta = log_z + sum_logpf - log_reward
    return delta ** 2


# ---------------------------------------------------------------------------
# Contrastive Trajectory Balance (this paper)
# ---------------------------------------------------------------------------

@dataclass
class CTBStats:
    """Aux statistics from :func:`ctb_loss` (useful for logging / debugging)."""
    is_connected: float = 1.0
    num_components: float = 1.0
    masked_pair_ratio: float = 0.0


def _ngp_mask(reward_diff_abs: torch.Tensor, sigma: float) -> torch.Tensor:
    """Noisy Gradient Pruning mask: True where |Δlog R| > sigma."""
    if sigma <= 0.0:
        return torch.ones_like(reward_diff_abs, dtype=torch.bool)
    return reward_diff_abs >= sigma


def _pair_weights(
    reward_diff_abs: torch.Tensor, pair_weight_temp: float
) -> torch.Tensor:
    """sigmoid((|Δlog R| - 0.5) / temp) — emphasizes high-information pairs.

    Setting ``pair_weight_temp`` very large (≥1e3) recovers near-uniform weights,
    which is the regime reported in the paper.
    """
    return torch.sigmoid((reward_diff_abs - 0.5) / pair_weight_temp)


def ctb_loss(
    sum_logpf: torch.Tensor,
    log_reward: torch.Tensor,
    *,
    mode: BaselineMode = "all_pairs",
    sigma: float = 0.0,
    pair_weight_temp: float = 100.0,
    c_log_reward: Optional[torch.Tensor] = None,
    min_pairwise_c_reward: Optional[float] = None,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, CTBStats]:
    """Contrastive Trajectory Balance loss.

    For every pair ``(i, j)``, the per-pair loss is
    ``huber((log PF_i - log PF_j) - (log R_i - log R_j))``.

    Args:
      sum_logpf: ``[B]`` total log-probability of each trajectory under π_θ.
      log_reward: ``[B]`` log-reward of each trajectory.
      mode:
        - ``"all_pairs"`` (default): every (i,j) with i<j (paper).
        - ``"mean_baseline"``: each sample vs. batch mean (cheaper).
        - ``"median_baseline"``: each sample vs. batch median.
      sigma: NGP threshold. Pairs whose absolute log-reward difference is
        below this are masked out. ``0.0`` disables NGP.
      pair_weight_temp: temperature of the sigmoid weighting on each pair.
        Use a very large value (≈100) to approximate uniform weighting.
      c_log_reward: optional classifier-only reward, only used by the optional
        ``min_pairwise_c_reward`` filter.
      min_pairwise_c_reward: if set, any pair where BOTH samples have
        ``c_log_reward`` below this threshold is dropped. Useful early in
        training when most samples are non-toxic.
      return_stats: if True, also return a :class:`CTBStats` summarizing
        graph connectivity of the unmasked pair graph (slow; use for debug).

    Returns:
      ``(losses, stats)`` where ``losses`` is a 1-D tensor of pair losses
      (its mean is the scalar objective).
    """
    B = sum_logpf.size(0)
    stats = CTBStats()

    if mode in ("mean_baseline", "median_baseline"):
        if mode == "mean_baseline":
            base_logpf = sum_logpf.mean()
            base_reward = log_reward.mean()
        else:
            idx = int(torch.median(log_reward, dim=0).indices.item())
            base_logpf = sum_logpf[idx]
            base_reward = log_reward[idx]

        logpf_diff = sum_logpf - base_logpf
        reward_diff = log_reward - base_reward
        delta = logpf_diff - reward_diff
        losses = F.huber_loss(
            delta, torch.zeros_like(delta), delta=1.0, reduction="none"
        )

        reward_diff_abs = torch.abs(reward_diff)
        mask = _ngp_mask(reward_diff_abs, sigma)
        weights = _pair_weights(reward_diff_abs, pair_weight_temp)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return losses * weights, stats

    # ----- all_pairs (paper default) -----
    logpf_diff = sum_logpf.unsqueeze(1) - sum_logpf.unsqueeze(0)  # [B,B]
    reward_diff = log_reward.unsqueeze(1) - log_reward.unsqueeze(0)  # [B,B]
    delta = logpf_diff - reward_diff
    pair_losses = F.huber_loss(
        delta, torch.zeros_like(delta), delta=1.0, reduction="none"
    )

    reward_diff_abs = torch.abs(reward_diff)
    mask = _ngp_mask(reward_diff_abs, sigma)

    if c_log_reward is not None and min_pairwise_c_reward is not None:
        below = c_log_reward < min_pairwise_c_reward  # [B]
        both_below = below.unsqueeze(1) & below.unsqueeze(0)
        mask = mask & ~both_below

    weights = _pair_weights(reward_diff_abs, pair_weight_temp)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weighted = pair_losses * weights

    # Upper-triangular flatten (avoid double-counting (i,j) and (j,i)).
    upper = torch.triu(
        torch.ones(B, B, dtype=torch.bool, device=sum_logpf.device), diagonal=1
    )
    losses = weighted[upper]

    if return_stats:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:  # pragma: no cover
            return losses, stats

        adj = mask.cpu().numpy().astype(int)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(csgraph=csr_matrix(adj), directed=False)
        total_pairs = B * (B - 1) / 2.0
        connected_pairs = int(mask.cpu().numpy()[upper.cpu().numpy()].sum())
        stats = CTBStats(
            is_connected=1.0 if n_comp == 1 else 0.0,
            num_components=float(n_comp),
            masked_pair_ratio=1.0 - (connected_pairs / total_pairs if total_pairs else 0.0),
        )

    return losses, stats
