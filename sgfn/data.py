"""Datasets and dataloaders used across S-GFN.

Three datasets are exposed:

* :class:`SFTDataset` — pairs the attacker meta-prompt with toxic instructions
  for the initial SFT warm-up.
* :class:`SafetyDataset` — instruction–response pairs used by the
  safety-fine-tuning baseline.
* :class:`RedTeamDataset` — streams attacker meta-prompts from a JSONL file
  during GFN training (one prompt per record).

The :func:`get_dataloader` helper builds the right dataloader for each mode.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import DataCollatorForSeq2Seq


# ---------------------------------------------------------------------------
# SFT (initial attacker warm-up)
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Meta-prompt + harmful instruction pairs, with a deterministic 90/10 split."""

    def __init__(
        self,
        tokenizer,
        prompt_file: str,
        instruction_file: str,
        split: str = "train",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.prompts: List[str] = []
        for line in Path(prompt_file).read_text().splitlines():
            if not line.strip():
                continue
            self.prompts.append(json.loads(line)["attacker_prompt"])

        instructions = json.loads(Path(instruction_file).read_text())
        instructions = [x["instruction"].strip() for x in instructions]
        random.seed(seed)
        random.shuffle(instructions)

        n_val = int(len(instructions) * 0.1)
        if split == "train":
            self.instructions = instructions[n_val:]
        elif split == "val":
            self.instructions = instructions[:n_val]
        elif split == "full":
            self.instructions = instructions
        else:
            raise ValueError(f"Unknown split: {split!r}")

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, index: int):
        prompt = random.choice(self.prompts)
        instruction = self.instructions[index]
        return self._encode(prompt, instruction)

    def _encode(self, prompt: str, instruction: str):
        full = prompt + " " + instruction
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        example_ids = self.tokenizer.encode(full) + [self.tokenizer.eos_token_id]
        example = torch.tensor(example_ids, dtype=torch.long)
        labels = copy.deepcopy(example)
        labels[: len(prompt_ids)] = -100
        attention_mask = example.ge(0)
        # Mask labels at padded positions (none here, but kept for consistency).
        labels[~attention_mask] = -100
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


# ---------------------------------------------------------------------------
# Safety FT (cross-attack defense)
# ---------------------------------------------------------------------------

class SafetyDataset(Dataset):
    """Instruction–response pairs for safety fine-tuning.

    If ``reweighting=True``, expects each record to carry a ``weight`` field
    that is used by a :class:`WeightedRandomSampler` at load time.
    """

    def __init__(self, tokenizer, instruction_file: str, reweighting: bool = False):
        self.tokenizer = tokenizer
        self.data = json.loads(Path(instruction_file).read_text())
        self.reweighting = reweighting
        self.weights: Optional[torch.Tensor] = None
        if reweighting:
            w = torch.tensor([item["weight"] for item in self.data], dtype=torch.float)
            self.weights = w / w.sum()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        return self._encode(item["instruction"], item["response"])

    def _encode(self, prompt: str, response: str):
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = chat_prompt + " " + response
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        example_ids = self.tokenizer.encode(full) + [self.tokenizer.eos_token_id]
        example = torch.tensor(example_ids, dtype=torch.long)
        labels = copy.deepcopy(example)
        labels[: len(prompt_ids)] = -100
        attention_mask = example.ge(0)
        labels[~attention_mask] = -100
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


# ---------------------------------------------------------------------------
# Red-team (GFN training)
# ---------------------------------------------------------------------------

class RedTeamDataset(Dataset):
    """Streams attacker meta-prompts from a JSONL file.

    The original code used ``subprocess + linecache`` for laziness. We load
    the (typically tiny) file once in-memory which is faster and avoids the
    OS-dependency.
    """

    def __init__(self, jsonl_file: str):
        self.prompts: List[str] = []
        for line in Path(jsonl_file).read_text().splitlines():
            if not line.strip():
                continue
            self.prompts.append(json.loads(line)["attacker_prompt"])

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    mode: str,
    tokenizer,
    prompt_file: str,
    sft_file: Optional[str] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    reweighting: bool = False,
):
    """Build the dataloader for ``mode`` ∈ ``{"gfn", "sft", "safety"}``.

    For ``"sft"`` two dataloaders are returned (train, val).
    """
    if mode == "gfn":
        ds = RedTeamDataset(prompt_file)

        def collate(batch):
            return tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        return DataLoader(ds, batch_size, shuffle=shuffle, collate_fn=collate)

    if mode == "sft":
        if sft_file is None:
            raise ValueError("sft_file is required for mode='sft'")
        collator = DataCollatorForSeq2Seq(tokenizer)
        tr = SFTDataset(tokenizer, prompt_file, sft_file, split="train")
        val = SFTDataset(tokenizer, prompt_file, sft_file, split="val")
        tr_dl = DataLoader(tr, batch_size, shuffle=shuffle, collate_fn=collator)
        val_dl = DataLoader(val, batch_size, shuffle=False, collate_fn=collator)
        return tr_dl, val_dl

    if mode == "safety":
        ds = SafetyDataset(tokenizer, prompt_file, reweighting=reweighting)
        collator = DataCollatorForSeq2Seq(tokenizer)
        if reweighting:
            sampler = WeightedRandomSampler(
                ds.weights, num_samples=len(ds), replacement=True
            )
            return DataLoader(ds, batch_size, collate_fn=collator, sampler=sampler)
        return DataLoader(ds, batch_size, shuffle=shuffle, collate_fn=collator)

    raise ValueError(f"Unknown dataloader mode: {mode!r}")
