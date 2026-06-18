"""Safety fine-tuning of the victim model.

Used to evaluate cross-attack defensive coverage: take a set of (attack_prompt,
refusal) pairs found by one red-teaming method, fine-tune the victim model
with LoRA on those pairs, then evaluate ASR of other attack methods against
the resulting defense model.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import wandb
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..config import SGFNConfig
from ..data import get_dataloader
from ..utils import InfIterator, get_decay_parameter_names


class SafetyTrainer:
    """LoRA safety FT on (instruction, refusal) pairs."""

    def __init__(self, cfg: SGFNConfig, reweighting: bool = False):
        self.cfg = cfg
        wandb.init(
            reinit=True,
            config=cfg.to_dict(),
            project=cfg.io.wandb_project,
            name=cfg.io.exp_name,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.victim_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=cfg.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.victim_name, padding_side="left"
        )
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

        self.dataloader = get_dataloader(
            "safety",
            self.tokenizer,
            prompt_file=cfg.io.prompt_file,
            batch_size=cfg.optim.batch_size,
            reweighting=reweighting,
        )
        self.train_iter = InfIterator(self.dataloader)

    def _save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def train(self) -> None:
        cfg = self.cfg
        t = tqdm(range(1, cfg.optim.train_steps + 1), desc="safety-ft", dynamic_ncols=True)
        for global_step in t:
            self.model.train()
            self.model.zero_grad()

            batch = next(self.train_iter)
            chunks = {k: torch.chunk(v, cfg.optim.grad_acc_steps, dim=0)
                      for k, v in batch.items()}
            n = len(chunks["input_ids"])
            step_losses = []
            for i in range(n):
                mini = {k: v[i].to(self.model.device) for k, v in chunks.items()}
                loss = self.model(**mini).loss / cfg.optim.grad_acc_steps
                loss.backward()
                step_losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            wandb.log({"ce-loss/train": sum(step_losses)}, step=global_step)
            t.set_description(f"step {global_step}: {sum(step_losses):.4f}")

        out = os.path.join(cfg.io.save_dir, cfg.io.exp_name, "latest")
        self._save(out)
        wandb.finish()
