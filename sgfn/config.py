"""Typed, YAML-backed configuration for S-GFN.

All trainers consume a :class:`SGFNConfig`. The config is composed of small,
self-describing dataclasses so users can either:

* edit a YAML file and load it with :func:`SGFNConfig.from_yaml`, or
* construct the config programmatically in Python.

CLI key=value overrides are supported via :func:`apply_overrides`.

Note: we deliberately keep YAML parsing dependency-free by using ``json`` for
nested dicts. If you prefer YAML, install ``pyyaml`` and replace the loader.
"""
from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# YAML is loaded lazily so it stays an optional dependency.
_yaml = None


def _maybe_yaml():
    global _yaml
    if _yaml is None:
        try:
            import yaml as _y
            _yaml = _y
        except ImportError:  # pragma: no cover
            _yaml = False
    return _yaml or None


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Attacker / victim / classifier model selection."""
    # Attacker (the policy being trained)
    name: str = "Qwen/Qwen2.5-1.5B"
    sft_ckpt: str = "save/qwen2.5-1.5b-sft/latest"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    # Victim (frozen, served via vLLM)
    victim_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    victim_top_p: float = 0.92
    victim_temp: float = 0.7
    victim_max_len: int = 30
    victim_max_model_len: Optional[int] = None
    victim_lora_path: str = ""
    num_response_samples: int = 5

    # Classifier (toxicity / safety judge)
    classifier: Literal["llama_guard", "harmaug"] = "llama_guard"
    classifier_model_id: str = "meta-llama/Llama-Guard-3-8B"


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 32
    alpha: int = 16
    dropout: float = 0.0


@dataclass
class GPUConfig:
    """GPU placement for the 4-process setup used in the paper."""
    attacker_devices: List[int] = field(default_factory=lambda: [3])
    victim_device: int = 0
    classifier_devices: List[int] = field(default_factory=lambda: [1, 2])
    attacker_gpu_memory_utilization: float = 0.8
    victim_gpu_memory_utilization: float = 0.85
    classifier_gpu_memory_utilization: float = 0.85


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_warmup_steps: int = 200
    train_steps: int = 600
    grad_acc_steps: int = 2
    batch_size: int = 12  # Effective on-policy batch *per* grad-acc step.


@dataclass
class GenerationConfig:
    min_len: int = 5
    max_len: int = 20
    sample_temperature: float = 1.0
    # On-policy temperature mixing (paper default): with probability
    # ``temp_mix_prob`` draw temp ~ Uniform[temp_low, temp_high], otherwise
    # use ``sample_temperature``. Required to prevent mode collapse.
    temp_low: float = 0.5
    temp_high: float = 2.0
    temp_mix_prob: float = 0.5
    logpf_aggregation: Literal["sum", "mean"] = "sum"


@dataclass
class BufferConfig:
    metric: Literal["edit", "cosine"] = "edit"
    size: int = 1000
    sim_tolerance: float = 0.4
    prioritization: Literal["reward", "c_reward", "uniform"] = "c_reward"
    compare: Literal["reward", "c_reward"] = "reward"
    min_reward: float = -25.0
    min_c_reward: float = -2.5
    paired_sampling: bool = True
    paired_on_policy_ratio: float = 0.67
    buffer_sample_prob: float = 0.5  # used only when paired_sampling=False
    buffer_path: str = ""             # optional pre-initialized buffer
    init_extra_samples: int = 10
    init_max_sft_samples: int = 300   # cap for buffer.init_buffer_from_sft seed


@dataclass
class LossConfig:
    """Which GFN loss to use, plus its hyper-parameters.

    Set ``kind="ctb"`` (default) to use Contrastive Trajectory Balance (the
    paper's loss). Set ``kind="tb"`` to use the original Trajectory Balance
    baseline.
    """
    kind: Literal["ctb", "tb"] = "ctb"
    # CTB-specific
    baseline_mode: Literal["all_pairs", "mean_baseline", "median_baseline"] = "all_pairs"
    pair_weight_temp: float = 100.0
    # TB-specific
    log_z_bias: float = 0.0


@dataclass
class NGPConfig:
    """Noisy Gradient Pruning settings (CTB only)."""
    sigma: float = 0.1
    sched_start: Optional[float] = None
    sched_end: Optional[float] = None
    sched_horizon: int = 500
    sched_kind: Literal["constant", "linear", "step"] = "constant"


@dataclass
class RewardConfig:
    """Reward composition and reward-side filtering."""
    # Min-K Fluency Stabilizer (MKS)
    lm_aggregation: Literal["sum", "mean", "bottom_k_mean"] = "bottom_k_mean"
    lm_bottom_k: int = 7

    # Annealing temperatures
    beta: float = 0.1                        # classifier-reward scale
    reward_sched_start: float = 1.0
    reward_sched_end: float = 1.0
    reward_sched_horizon: int = 500
    lm_sched_start: float = 1.0
    lm_sched_end: float = 1.0
    lm_sched_horizon: int = 2000

    # Hard clipping on the LM reward
    lm_clip_enabled: bool = True
    lm_clip_threshold: float = -10.0
    lm_clip_value: float = -300.0

    # Set True to drop the LM reward term entirely (classifier reward only).
    disable_lm_reward: bool = False


@dataclass
class IOConfig:
    prompt_file: str = "prompts/attack_prompt.jsonl"
    few_shot_file: str = "prompts/sft_dataset.json"
    log_dir: str = "logs"
    save_dir: str = "save"
    save_interval: int = 100
    exp_name: str = "sgfn"
    wandb_project: str = "sgfn"
    resume: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class SGFNConfig:
    """All hyper-parameters needed to run any S-GFN trainer."""
    mode: Literal["sft", "redteam", "safety"] = "redteam"
    seed: int = 42
    debug: bool = False

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ngp: NGPConfig = field(default_factory=NGPConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # -- loaders --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SGFNConfig":
        text = Path(path).read_text()
        yaml = _maybe_yaml()
        if yaml is None:
            # Fall back to JSON (YAML is a superset; pure JSON files work).
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SGFNConfig":
        return _from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# (de)serialization helpers
# ---------------------------------------------------------------------------

def _from_dict(cls, data: Dict[str, Any]):
    if not is_dataclass(cls):
        return data
    # ``f.type`` is a string under ``from __future__ import annotations``;
    # resolve type hints to actual classes for nested-dataclass detection.
    type_hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        ftype = type_hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[f.name] = _from_dict(ftype, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _to_dict(obj) -> Dict[str, Any]:
    if not is_dataclass(obj):
        return obj
    out: Dict[str, Any] = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        out[f.name] = _to_dict(v) if is_dataclass(v) else v
    return out


def apply_overrides(cfg: SGFNConfig, overrides: List[str]) -> SGFNConfig:
    """Apply ``dotted.path=value`` overrides parsed from the CLI.

    Values are parsed as JSON when possible (``true``, ``1.5``, ``[1,2]``);
    they fall back to raw strings otherwise.
    """
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Bad override '{o}' (expected key=value).")
        key, raw = o.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_nested(cfg, key.split("."), value)
    return cfg


def _set_nested(obj, path: List[str], value) -> None:
    cur = obj
    for p in path[:-1]:
        cur = getattr(cur, p)
    setattr(cur, path[-1], value)
