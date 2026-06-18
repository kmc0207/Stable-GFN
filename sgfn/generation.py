"""On-policy generation with per-token log-probabilities and a log-Z head.

This is the autoregressive sampler used by the GFlowNet trainer. It returns:

* ``actions``       — the full input (prompt + response + EOS), right-padded.
* ``sum_logpf``     — Σ log π_θ(y_t | y_<t), or its length-mean (configurable).
* ``log_z``         — output of the ``proj_z`` head on the prompt encoding; kept
  only for logging when training with CTB (CTB does not require log-Z).

The function is deliberately framework-agnostic so it can be reused outside
this trainer; the only requirement is that ``model`` has an attached
``proj_z`` linear head (see :func:`attach_log_z_head`).
"""
from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


LogPfAggregation = Literal["sum", "mean"]


def attach_log_z_head(model, hidden_size: int, device: torch.device | str) -> None:
    """Attach a single linear ``proj_z`` head used to estimate log-Z(s).

    Only used by standard TB; CTB does not back-propagate through this head,
    but we keep it attached so checkpoints stay forward-compatible.
    """
    model.proj_z = nn.Linear(hidden_size, 1).to(device)


def _avg_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_f = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    denom = torch.clamp(mask_f.sum(1), min=1)
    return torch.sum(last_hidden * mask_f, 1) / denom


@torch.no_grad()
def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    prob = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(prob, num_samples=1)


def generate_with_logprobs(
    model,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    eos_token_id: int,
    *,
    temperature: float,
    max_len: int = 30,
    logpf_aggregation: LogPfAggregation = "sum",
) -> Dict[str, torch.Tensor]:
    """Autoregressive sampling from ``model`` while tracking ``log π_θ``.

    Args:
      model: a CausalLM with a ``proj_z`` linear head attached. The model is
        expected to be in train mode (no ``torch.no_grad``).
      prompt_ids: ``[B, P]`` left-padded prompt ids.
      prompt_attention_mask: ``[B, P]`` attention mask for the prompt.
      eos_token_id: EOS id; used to stop and to pad finished sequences.
      temperature: sampling temperature (>0).
      max_len: maximum number of new tokens.
      logpf_aggregation: ``"sum"`` (default, matches the paper) or ``"mean"``.

    Returns:
      Dict with keys ``actions``, ``sum_logpf``, ``log_z``.
    """
    device = prompt_ids.device
    B = prompt_ids.size(0)
    active = torch.ones(B, dtype=torch.bool, device=device)
    state = prompt_ids.clone()
    attn = prompt_attention_mask.clone()
    sum_logpf = torch.zeros(B, dtype=torch.float, device=device)
    gen_lengths = torch.zeros(B, dtype=torch.long, device=device)

    # Initial pass on prompt[:, :-1] to populate KV cache and pool hidden states.
    out = model(
        input_ids=state[:, :-1],
        attention_mask=attn[:, :-1],
        output_hidden_states=True,
        use_cache=True,
    )
    hidden_states_seq = out.hidden_states[-1]
    past_kv = out.past_key_values

    log_z = None
    for step in range(max_len):
        inputs = {
            "input_ids": state[:, -1:],
            "attention_mask": attn,
            "past_key_values": past_kv,
        }
        if step == 0:
            inputs["output_hidden_states"] = True
            inputs["use_cache"] = True
            out = model(**inputs)
            last_hidden = out.hidden_states[-1]
            full_hidden = torch.cat([hidden_states_seq, last_hidden], dim=1)
            pooled = _avg_pool(full_hidden, prompt_attention_mask)
            log_z = model.proj_z(pooled).squeeze(-1)
        else:
            out = model(**inputs)
        past_kv = getattr(out, "past_key_values", None)
        logits = out.logits[:, -1, :]
        if step == 0:
            logits[..., eos_token_id] = -float("inf")

        token_ids = _sample_token(logits, temperature)
        logprob = F.log_softmax(logits, dim=-1).gather(-1, token_ids).squeeze(-1)
        logprob = torch.where(active, logprob, torch.zeros_like(logprob))
        sum_logpf = sum_logpf + logprob
        gen_lengths = gen_lengths + active.long()

        token_ids = torch.where(
            active.unsqueeze(-1), token_ids, torch.full_like(token_ids, eos_token_id)
        )
        new_mask = torch.where(
            active.unsqueeze(-1),
            torch.ones_like(token_ids),
            torch.zeros_like(token_ids),
        )
        attn = torch.cat([attn.long(), new_mask], dim=1)
        state = torch.cat([state, token_ids], dim=1)
        active = active & (token_ids != eos_token_id).squeeze(-1)
        if not active.any():
            break

    if logpf_aggregation == "mean":
        sum_logpf = sum_logpf / torch.clamp(gen_lengths.float(), min=1.0)

    # Append a final EOS so downstream code can treat sequences uniformly.
    eos_col = torch.full((B, 1), eos_token_id, dtype=torch.long, device=device)
    actions = torch.cat([state, eos_col], dim=1)

    return {"actions": actions, "sum_logpf": sum_logpf, "log_z": log_z}
