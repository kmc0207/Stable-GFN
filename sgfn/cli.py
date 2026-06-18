"""Command-line interface for S-GFN.

Usage::

    # Run S-GFN training with a YAML config
    sgfn train --config configs/sgfn.yaml

    # Override config keys from the CLI (JSON-parsed values)
    sgfn train --config configs/sgfn.yaml --set optim.lr=2e-4 buffer.paired_on_policy_ratio=0.5

    # Run SFT warm-up
    sgfn train --config configs/sft.yaml

    # Run safety fine-tuning of a victim model
    sgfn train --config configs/safety.yaml

The ``--set`` flag accepts ``dotted.key=value`` pairs; values are parsed as
JSON when possible (``true``, ``1.5``, ``"path/to/x"``) and as raw strings
otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .config import SGFNConfig, apply_overrides
from .utils import set_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgfn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run a trainer (SFT, S-GFN, or safety FT).")
    train.add_argument("--config", required=True, type=str, help="Path to a YAML/JSON config file.")
    train.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="key=value",
        help="Inline dotted overrides applied on top of the config file.",
    )

    show = sub.add_parser("show-config", help="Print the resolved config and exit.")
    show.add_argument("--config", required=True, type=str)
    show.add_argument("--set", nargs="*", default=[])

    eval_p = sub.add_parser("eval", help="Evaluate a trained attacker (ASR + UA).")
    eval_p.add_argument("--config", required=True, type=str,
                        help="Eval YAML (uses the same SGFNConfig schema).")
    eval_p.add_argument("--ckpt", required=True, type=str,
                        help="Attacker checkpoint directory (LoRA adapter or full).")
    eval_p.add_argument("--output-dir", required=True, type=str,
                        help="Where to write attacks.json / scored.json / metrics.json.")
    eval_p.add_argument("--num-samples", type=int, default=1024)
    eval_p.add_argument("--attacker-batch-size", type=int, default=32)
    eval_p.add_argument("--victim-batch-size", type=int, default=16)
    eval_p.add_argument("--num-responses-per-attack", type=int, default=5)
    eval_p.add_argument("--set", nargs="*", default=[])

    return parser


def _load_config(path: str, overrides: List[str]) -> SGFNConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = SGFNConfig.from_yaml(path)
    return apply_overrides(cfg, overrides)


def _run_train(cfg: SGFNConfig) -> None:
    set_seed(cfg.seed)
    # Lazy import so `--help` / `show-config` don't require torch+vllm.
    if cfg.mode == "sft":
        from .trainers import SFTTrainer
        SFTTrainer(cfg).train()
    elif cfg.mode == "redteam":
        from .trainers import GFNTrainer
        GFNTrainer(cfg).train()
    elif cfg.mode == "safety":
        from .trainers import SafetyTrainer
        SafetyTrainer(cfg).train()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}")


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        cfg = _load_config(args.config, args.set)
        _run_train(cfg)
        return 0

    if args.command == "show-config":
        import json
        cfg = _load_config(args.config, args.set)
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    if args.command == "eval":
        cfg = _load_config(args.config, args.set)
        set_seed(cfg.seed)
        from .eval import run_eval
        run_eval(
            cfg,
            attacker_ckpt=args.ckpt,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            attacker_batch_size=args.attacker_batch_size,
            victim_batch_size=args.victim_batch_size,
            num_responses_per_attack=args.num_responses_per_attack,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
