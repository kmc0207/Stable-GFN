"""Small helpers shared across trainers."""
from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class InfIterator:
    """Wrap a finite iterable into an infinite iterator (loops on StopIteration)."""

    def __init__(self, iterable: Iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


def formatted_dict(d: Dict) -> Dict:
    """Format floats in a dict for compact stdout printing."""
    return {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in d.items()}


# ---------- LoRA helpers ----------

def lora_to_base(model) -> None:
    """Disable LoRA adapters so the wrapped model behaves as the base model."""
    try:
        model.base_model.disable_adapter_layers()
    except AttributeError:
        pass
    model.eval()


def base_to_lora(model) -> None:
    """Re-enable LoRA adapters and put the model in training mode."""
    try:
        model.base_model.enable_adapter_layers()
    except AttributeError:
        pass
    model.train()


# ---------- Optimizer helpers ----------

def _parameter_names(model, forbidden_layer_types) -> List[str]:
    result: List[str] = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in _parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """Parameter names eligible for weight decay (excludes LayerNorm + biases)."""
    decay = _parameter_names(model, ALL_LAYERNORM_LAYERS)
    return [n for n in decay if "bias" not in n]
