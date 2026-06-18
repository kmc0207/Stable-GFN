"""Initial SFT warm-up for the attacker model.

The SFT data is the AdvBench / safety toxic-instruction set; the model learns
to *produce* harmful one-line instructions, which is the starting point of
the red-teaming pipeline.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..config import SGFNConfig
from ..data import get_dataloader
from ..utils import InfIterator, get_decay_parameter_names


def _get_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that respect left-padding (transformers convention)."""
    pos = attention_mask.long().cumsum(-1) - 1
    pos.masked_fill_(attention_mask == 0, 1)
    return pos


class SFTTrainer:
    """Standard SFT on left-padded instruction inputs."""

    def __init__(self, cfg: SGFNConfig):
        self.cfg = cfg
        wandb.init(
            reinit=True,
            config=cfg.to_dict(),
            project=cfg.io.wandb_project,
            name=cfg.io.exp_name,
        )

        self.device = torch.cuda.current_device()

        config = AutoConfig.from_pretrained(cfg.model.name)
        config.use_cache = True

        torch_dtype = torch.bfloat16 if cfg.lora.enabled else None
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=torch_dtype,
            config=config,
            device_map="auto",
        )

        if cfg.lora.enabled:
            lora_config = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        decay = get_decay_parameter_names(self.model)
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if n in decay and p.requires_grad],
                "weight_decay": cfg.optim.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if n not in decay and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.optim.num_warmup_steps, cfg.optim.train_steps
        )

        self.train_dl, self.val_dl = get_dataloader(
            "sft",
            self.tokenizer,
            prompt_file=cfg.io.prompt_file,
            sft_file=cfg.io.few_shot_file,
            batch_size=cfg.optim.batch_size,
        )
        self.train_iter = InfIterator(self.train_dl)

    # ------------------------------------------------------------------

    def _validate(self) -> float:
        self.model.eval()
        losses = []
        for batch in tqdm(self.val_dl, leave=False, dynamic_ncols=True, desc="val"):
            chunks = {k: torch.chunk(v, self.cfg.optim.grad_acc_steps, dim=0)
                      for k, v in batch.items()}
            n = len(chunks["input_ids"])
            for i in range(n):
                mini = {k: v[i].to(self.device) for k, v in chunks.items()}
                with torch.no_grad():
                    pos = _get_position_ids(mini["attention_mask"])
                    losses.append(
                        self.model(**mini, position_ids=pos).loss.item()
                    )
        return float(np.mean(losses))

    def _save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self.cfg.lora.enabled:
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def train(self) -> None:
        cfg = self.cfg
        t = tqdm(range(1, cfg.optim.train_steps + 1), desc="sft", dynamic_ncols=True)
        for global_step in t:
            self.model.train()
            self.model.zero_grad()
            batch = next(self.train_iter)
            chunks = {k: torch.chunk(v, cfg.optim.grad_acc_steps, dim=0)
                      for k, v in batch.items()}
            n = len(chunks["input_ids"])
            step_losses = []
            for i in range(n):
                mini = {k: v[i].to(self.device) for k, v in chunks.items()}
                pos = _get_position_ids(mini["attention_mask"])
                loss = self.model(**mini, position_ids=pos).loss
                loss = loss / cfg.optim.grad_acc_steps
                loss.backward()
                step_losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            wandb.log({"ce-loss/train": sum(step_losses)}, step=global_step)
            t.set_description(f"step {global_step}: {sum(step_losses):.4f}")

            if global_step % 10 == 0:
                wandb.log({"ce-loss/val": self._validate()}, step=global_step)

        out = os.path.join(cfg.io.save_dir, cfg.io.exp_name, "latest")
        self._save(out)
        wandb.finish()
