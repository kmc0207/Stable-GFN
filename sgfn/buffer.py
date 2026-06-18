"""Replay buffer used to mix on-policy and off-policy trajectories.

Two flavors share the same API:
  - :class:`ReplayBuffer` deduplicates with token-level edit distance.
  - :class:`CosineReplayBuffer` deduplicates with cosine similarity over
    sentence embeddings. Cheaper at large buffer sizes.

When a near-duplicate of an incoming trajectory already lives in the buffer,
we keep whichever has the higher ``ref_reward`` (configurable via ``compare``).
"""
from __future__ import annotations

import gzip
import heapq
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Trajectory dataclasses
# ---------------------------------------------------------------------------

@dataclass(order=True)
class TrajectoryByReward:
    """Heap-ordered by the combined ``log_reward``."""
    response_ids: torch.Tensor = field(compare=False)
    c_log_reward: float = field(compare=False)
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=True)
    decoded_response: str = field(compare=False)
    emb: torch.Tensor = field(compare=False)
    prompt_ids: Optional[torch.Tensor] = field(compare=False, default=None)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.log_reward


@dataclass(order=True)
class TrajectoryByCReward:
    """Heap-ordered by classifier-only reward (``c_log_reward``)."""
    response_ids: torch.Tensor = field(compare=False)
    c_log_reward: float = field(compare=True)
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=False)
    decoded_response: str = field(compare=False)
    emb: torch.Tensor = field(compare=False)
    prompt_ids: Optional[torch.Tensor] = field(compare=False, default=None)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.c_log_reward


# ---------------------------------------------------------------------------
# Edit-distance buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Replay buffer with edit-distance based deduplication.

    Args:
      eos_token_id: token used to strip padding before computing edit distance.
      max_size: maximum number of trajectories kept; least-priority entries are
        evicted via ``heapq``.
      sim_tolerance: a trajectory is considered a duplicate of another if its
        edit distance is less than
        ``sim_tolerance * (len_a + len_b)``.
      prioritization: how sampling probabilities are computed at ``sample()``
        time. One of ``"reward"``, ``"c_reward"``, ``"uniform"``.
      compare: which reward orders the underlying heap; controls what wins
        ties on duplicates. One of ``"reward"`` (combined) or ``"c_reward"``.
      min_reward / min_c_reward: hard floors; incoming items below the floor
        are rejected.
    """

    def __init__(
        self,
        eos_token_id: int,
        max_size: int = 1000,
        sim_tolerance: float = 0.4,
        prioritization: str = "c_reward",
        compare: str = "reward",
        min_reward: float = -float("inf"),
        min_c_reward: float = -float("inf"),
    ):
        self.eos_token_id = eos_token_id
        self.max_size = max_size
        self.sim_tolerance = sim_tolerance
        self.prioritization = prioritization
        self.compare = compare
        self.min_reward = min_reward
        self.min_c_reward = min_c_reward

        self.buffer: List = []
        self.response_pool: set = set()
        self.Trajectory = (
            TrajectoryByCReward if compare == "c_reward" else TrajectoryByReward
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def size(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> Dict[str, float]:
        if not self.buffer:
            return {
                "buffer_size": 0,
                "buffer_c_reward_mean": 0.0,
                "buffer_c_reward_max": 0.0,
                "buffer_c_reward_min": 0.0,
                "buffer_lm_reward_mean": 0.0,
                "buffer_total_reward_mean": 0.0,
            }
        cs = [t.c_log_reward for t in self.buffer]
        lms = [t.lm_log_reward for t in self.buffer]
        totals = [t.log_reward for t in self.buffer]
        return {
            "buffer_size": len(self.buffer),
            "buffer_c_reward_mean": float(np.mean(cs)),
            "buffer_c_reward_max": float(np.max(cs)),
            "buffer_c_reward_min": float(np.min(cs)),
            "buffer_lm_reward_mean": float(np.mean(lms)),
            "buffer_total_reward_mean": float(np.mean(totals)),
        }

    # ------------------------------------------------------------------
    # Insertion / eviction
    # ------------------------------------------------------------------
    def _reward_floors_ok(self, item) -> bool:
        if item.log_reward < self.min_reward:
            return False
        if item.c_log_reward < self.min_c_reward:
            return False
        return True

    def _find_duplicate(self, tokens: List[int]):
        """Return (index, existing) of the first near-duplicate, else None."""
        for i, existing in enumerate(self.buffer):
            existing_tokens = [
                x for x in existing.response_ids.tolist() if x != self.eos_token_id
            ]
            denom = len(tokens) + len(existing_tokens)
            if denom == 0:
                continue
            dist = editdistance.eval(tokens, existing_tokens)
            if dist < denom * self.sim_tolerance:
                return i, existing
        return None

    def add(self, item) -> None:
        if not self._reward_floors_ok(item):
            return
        if item.decoded_response in self.response_pool:
            return

        tokens = [x for x in item.response_ids.tolist() if x != self.eos_token_id]
        dup = self._find_duplicate(tokens)
        if dup is not None:
            _, existing = dup
            if existing.ref_reward >= item.ref_reward:
                return  # existing item dominates
            # Replace existing duplicate with the new (higher-ref-reward) item.
            self.response_pool.discard(existing.decoded_response)
            self.buffer.remove(existing)
            heapq.heapify(self.buffer)
            self.response_pool.add(item.decoded_response)
            heapq.heappush(self.buffer, item)
            return

        # No duplicate -> insert normally.
        self.response_pool.add(item.decoded_response)
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.response_pool.discard(popped.decoded_response)
            except KeyError:
                self.response_pool = {t.decoded_response for t in self.buffer}

    def add_batch(
        self,
        responses: torch.Tensor,
        decoded_responses: List[str],
        res_embs: torch.Tensor,
        c_log_rewards: torch.Tensor,
        lm_log_rewards: torch.Tensor,
        log_rewards: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        responses = responses.cpu()
        res_embs = res_embs.cpu()
        if prompt_ids is not None:
            prompt_ids = prompt_ids.cpu()

        # Right-padded responses; strip trailing EOS for storage.
        pad_mask = (responses == self.eos_token_id).cumsum(1) > 1
        lengths = torch.sum((~pad_mask).long(), 1)

        for i in range(log_rewards.size(0)):
            response_id = responses[i, : lengths[i].item()]
            item = self.Trajectory(
                response_ids=response_id,
                c_log_reward=float(c_log_rewards[i].item()),
                lm_log_reward=float(lm_log_rewards[i].item()),
                log_reward=float(log_rewards[i].item()),
                decoded_response=decoded_responses[i],
                emb=res_embs[i],
                prompt_ids=prompt_ids[i] if prompt_ids is not None else None,
            )
            self.add(item)

        return self.get_stats()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sampling_probabilities(self) -> np.ndarray:
        if self.prioritization == "uniform":
            return np.ones(len(self.buffer)) / len(self.buffer)
        if self.prioritization == "reward":
            priorities = np.array([t.log_reward for t in self.buffer])
        elif self.prioritization == "c_reward":
            priorities = np.array([t.c_log_reward for t in self.buffer])
        else:
            raise ValueError(f"Unknown prioritization: {self.prioritization}")
        # softmax(priorities - max) for numerical stability
        priorities = priorities - priorities.max()
        priorities = np.exp(priorities)
        return priorities / priorities.sum()

    def sample(
        self, num_samples: int, return_embs: bool = False
    ) -> Tuple[Optional[Dict], Dict, Dict] | Tuple[Optional[Dict], Dict, Dict, torch.Tensor]:
        prob = self._sampling_probabilities()
        idx = np.random.choice(len(self.buffer), num_samples, p=prob, replace=False)

        response_ids = [self.buffer[i].response_ids for i in idx]
        response_mask = [torch.ones_like(x) for x in response_ids]
        response_ids = pad_sequence(
            response_ids, batch_first=True, padding_value=self.eos_token_id
        )
        response_mask = pad_sequence(
            response_mask, batch_first=True, padding_value=0
        )
        response_batch = {"input_ids": response_ids, "attention_mask": response_mask}

        # Left-padded prompts (if any were stored).
        prompt_ids_list = [self.buffer[i].prompt_ids for i in idx]
        if any(p is not None and len(p) > 0 for p in prompt_ids_list):
            rev = [
                torch.flip(p if p is not None else torch.tensor([], dtype=torch.long), [0])
                for p in prompt_ids_list
            ]
            rev_mask = [torch.ones_like(p) for p in rev]
            prompt_ids_pad = pad_sequence(
                rev, batch_first=True, padding_value=self.eos_token_id
            )
            prompt_mask_pad = pad_sequence(rev_mask, batch_first=True, padding_value=0)
            prompt_batch = {
                "input_ids": torch.flip(prompt_ids_pad, [1]),
                "attention_mask": torch.flip(prompt_mask_pad, [1]),
            }
        else:
            prompt_batch = None

        reward_batch = {
            "c_log_reward": torch.tensor([self.buffer[i].c_log_reward for i in idx]),
            "lm_log_reward": torch.tensor([self.buffer[i].lm_log_reward for i in idx]),
            "log_reward": torch.tensor([self.buffer[i].log_reward for i in idx]),
        }

        if return_embs:
            embs = torch.stack([self.buffer[i].emb for i in idx], dim=0)
            return prompt_batch, response_batch, reward_batch, embs
        return prompt_batch, response_batch, reward_batch

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path: str) -> None:
        with gzip.open(path, "rb") as f:
            self.buffer = pickle.load(f)
        heapq.heapify(self.buffer)
        self.response_pool = {t.decoded_response for t in self.buffer}

    def merge(self, other: "ReplayBuffer") -> None:
        self.max_size += other.max_size
        for item in other.buffer:
            self.add(item)


# ---------------------------------------------------------------------------
# Cosine-similarity buffer
# ---------------------------------------------------------------------------

class CosineReplayBuffer(ReplayBuffer):
    """Replay buffer that uses cosine similarity instead of edit distance.

    Pass a ``sim_tolerance`` in cosine units (typical: 0.4–0.6).
    """

    def __init__(
        self,
        eos_token_id: int,
        max_size: int = 1000,
        sim_tolerance: float = 0.4,
        prioritization: str = "c_reward",
        compare: str = "reward",
        min_reward: float = -float("inf"),
        min_c_reward: float = -float("inf"),
    ):
        super().__init__(
            eos_token_id,
            max_size,
            sim_tolerance,
            prioritization,
            compare,
            min_reward,
            min_c_reward,
        )

    def add(self, item) -> None:  # type: ignore[override]
        if not self._reward_floors_ok(item):
            return
        if item.decoded_response in self.response_pool:
            return

        if self.buffer:
            buf_embs = torch.stack([t.emb for t in self.buffer], dim=0)
            sims = F.cosine_similarity(item.emb.unsqueeze(0), buf_embs, dim=1)
            max_idx = int(torch.argmax(sims).item())
            if sims[max_idx].item() > self.sim_tolerance:
                existing = self.buffer[max_idx]
                if existing.ref_reward >= item.ref_reward:
                    return
                self.response_pool.discard(existing.decoded_response)
                self.buffer.remove(existing)
                heapq.heapify(self.buffer)
                self.response_pool.add(item.decoded_response)
                heapq.heappush(self.buffer, item)
                return

        self.response_pool.add(item.decoded_response)
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.response_pool.discard(popped.decoded_response)
            except KeyError:
                self.response_pool = {t.decoded_response for t in self.buffer}


def build_replay_buffer(
    metric: str,
    eos_token_id: int,
    max_size: int,
    sim_tolerance: float,
    prioritization: str,
    compare: str,
    min_reward: float = -float("inf"),
    min_c_reward: float = -float("inf"),
) -> ReplayBuffer:
    """Factory for the two replay buffer flavors."""
    if metric == "edit":
        cls = ReplayBuffer
    elif metric == "cosine":
        cls = CosineReplayBuffer
    else:
        raise ValueError(f"Unknown buffer metric: {metric!r} (expected 'edit'|'cosine')")
    return cls(
        eos_token_id=eos_token_id,
        max_size=max_size,
        sim_tolerance=sim_tolerance,
        prioritization=prioritization,
        compare=compare,
        min_reward=min_reward,
        min_c_reward=min_c_reward,
    )
