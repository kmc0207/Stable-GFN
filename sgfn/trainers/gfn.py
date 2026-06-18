"""S-GFN (Stable-GFlowNet) trainer.

Implements the training loop described in the paper:

  * a frozen victim LLM served via vLLM,
  * a frozen toxicity classifier (Llama-Guard by default),
  * a trainable attacker policy with a small ``proj_z`` head,
  * a replay buffer of high-reward attacks,
  * paired on-policy / off-policy sampling each gradient step,
  * **Contrastive Trajectory Balance (CTB)** loss with **Noisy Gradient
    Pruning (NGP)**,
  * **Min-K Fluency Stabilizer (MKS)** on the LM-reward.

The class is intentionally split into small methods so each piece is easy
to replace independently.

GPU placement note
------------------
This trainer expects 3 GPUs minimum (attacker / victim / classifier). The
victim model is loaded *first* (vLLM grabs a CUDA context) and only after
that does the trainer set up the attacker model and wandb. Trying to do
this in a different order frequently produces obscure CUDA errors.
"""
from __future__ import annotations

import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..buffer import build_replay_buffer
from ..classifiers import HarmAugClassifier, LlamaGuardClassifier
from ..config import SGFNConfig
from ..data import get_dataloader
from ..generation import attach_log_z_head, generate_with_logprobs
from ..losses import ctb_loss, tb_loss
from ..rewards import LMRewardClip, LinearScheduler, NGPScheduler, aggregate_lm_logreward
from ..utils import (
    InfIterator,
    base_to_lora,
    formatted_dict,
    lora_to_base,
)


# Suppress noisy decoder-only padding warnings from transformers
import warnings
warnings.filterwarnings("ignore", message=".*padding_side.*decoder-only.*")
warnings.filterwarnings("ignore", message=".*padding_side.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _avg_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return torch.sum(hidden * m, 1) / torch.clamp(m.sum(1), min=1)


def _make_chat_prompt(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction.rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _hidden_size(model_config) -> int:
    # Gemma models nest their text config; everything else has a flat attribute.
    if hasattr(model_config, "text_config"):
        return model_config.text_config.hidden_size
    return model_config.hidden_size


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GFNTrainer:
    """Stable-GFlowNet trainer for LLM red-teaming."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg: SGFNConfig):
        self.cfg = cfg

        # ---- 1. Print + set up GPU plan (do NOT init CUDA yet) ----
        print(
            f"GPU plan: attacker={cfg.gpu.attacker_devices}, "
            f"victim={cfg.gpu.victim_device}, classifier={cfg.gpu.classifier_devices}"
        )

        # ---- 2. Victim model (vLLM) MUST come before anything CUDA-touching ----
        self._init_victim()

        # ---- 3. wandb ----
        wandb.init(
            reinit=True,
            config=cfg.to_dict(),
            project=cfg.io.wandb_project,
            name=cfg.io.exp_name,
        )

        # ---- 4. Attacker policy ----
        self.device = torch.device(f"cuda:{cfg.gpu.attacker_devices[0]}")
        torch.cuda.set_device(self.device)
        self._init_attacker()

        # ---- 5. Toxicity classifier ----
        self._init_classifier()

        # ---- 6. Sentence encoder (used for buffer dedup + diversity stats) ----
        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.device
        )

        # ---- 7. Dataloader + optimizer ----
        self.dataloader = get_dataloader(
            "gfn",
            self.tokenizer,
            prompt_file=cfg.io.prompt_file,
            batch_size=cfg.optim.batch_size,
            shuffle=True,
        )
        self.train_iter = InfIterator(self.dataloader)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.optim.lr)
        t_total = cfg.optim.train_steps * cfg.optim.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.optim.num_warmup_steps, t_total
        )

        # ---- 8. Replay buffer ----
        self.rbuffer = build_replay_buffer(
            metric=cfg.buffer.metric,
            eos_token_id=self.tokenizer.eos_token_id,
            max_size=cfg.buffer.size,
            sim_tolerance=cfg.buffer.sim_tolerance,
            prioritization=cfg.buffer.prioritization,
            compare=cfg.buffer.compare,
            min_reward=cfg.buffer.min_reward,
            min_c_reward=cfg.buffer.min_c_reward,
        )

        # ---- 9. Reward processing helpers ----
        self.lm_temp = LinearScheduler(
            start=cfg.reward.lm_sched_start,
            end=cfg.reward.lm_sched_end,
            horizon=cfg.reward.lm_sched_horizon,
        )
        self.reward_temp = LinearScheduler(
            start=cfg.reward.reward_sched_start,
            end=cfg.reward.reward_sched_end,
            horizon=cfg.reward.reward_sched_horizon,
        )
        self.lm_clip = LMRewardClip(
            threshold=cfg.reward.lm_clip_threshold,
            clipped_value=cfg.reward.lm_clip_value,
            enabled=cfg.reward.lm_clip_enabled,
        )
        self.ngp = NGPScheduler(
            base=cfg.ngp.sigma,
            start=cfg.ngp.sched_start,
            end=cfg.ngp.sched_end,
            horizon=cfg.ngp.sched_horizon,
            kind=cfg.ngp.sched_kind,
        )

        # ---- 10. Maybe resume ----
        self.start_step = self._maybe_resume()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_victim(self) -> None:
        cfg = self.cfg
        kwargs = {
            "dtype": cfg.model.dtype,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": cfg.gpu.victim_gpu_memory_utilization,
        }
        if cfg.model.victim_max_model_len is not None:
            kwargs["max_model_len"] = cfg.model.victim_max_model_len

        enable_lora = bool(cfg.model.victim_lora_path)
        if enable_lora:
            self.victim_model = LLM(
                cfg.model.victim_name,
                enable_lora=True,
                max_loras=4,
                max_lora_rank=64,
                **kwargs,
            )
        else:
            self.victim_model = LLM(cfg.model.victim_name, **kwargs)
        self.victim_tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.victim_name, padding_side="left"
        )
        self.victim_tokenizer.pad_token_id = self.victim_tokenizer.eos_token_id
        self.sampling_params = SamplingParams(
            n=cfg.model.num_response_samples,
            top_p=cfg.model.victim_top_p,
            temperature=cfg.model.victim_temp,
            stop_token_ids=[self.victim_tokenizer.eos_token_id],
            max_tokens=cfg.model.victim_max_len,
        )

    def _init_attacker(self) -> None:
        cfg = self.cfg
        config = AutoConfig.from_pretrained(cfg.model.name)
        config.use_cache = True
        dtype = _torch_dtype(cfg.model.dtype)

        if len(cfg.gpu.attacker_devices) > 1:
            max_memory = {i: "0GiB" for i in range(torch.cuda.device_count())}
            for gid in cfg.gpu.attacker_devices:
                max_memory[gid] = "22GiB"
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.sft_ckpt,
                config=config,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.sft_ckpt,
                config=config,
                torch_dtype=dtype,
                device_map={"": self.device},
            )

        if cfg.lora.enabled:
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

        attach_log_z_head(self.model, _hidden_size(self.model.config), self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.sft_ckpt, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _init_classifier(self) -> None:
        cfg = self.cfg
        if cfg.model.classifier == "llama_guard":
            self.toxicity_fn = LlamaGuardClassifier(
                model_id=cfg.model.classifier_model_id,
                device_ids=cfg.gpu.classifier_devices,
                dtype=_torch_dtype(cfg.model.dtype),
            )
        elif cfg.model.classifier == "harmaug":
            device = torch.device(f"cuda:{cfg.gpu.classifier_devices[0]}")
            self.toxicity_fn = HarmAugClassifier(device)
        else:
            raise ValueError(f"Unknown classifier: {cfg.model.classifier!r}")

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def _lm_logreward_from_reference(
        self, prompts_responses: torch.Tensor, prompt_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample LM log-reward under the *reference* (base) model + MKS.

        Returns ``(lm_logreward, pad_mask)`` where ``pad_mask`` is ``[B, T_resp]``
        marking padding positions in the response part.
        """
        only_responses = prompts_responses[:, prompt_len:]
        pad_mask = (only_responses == self.tokenizer.pad_token_id).cumsum(1) > 1

        # Build the attention mask for the full prompt+response sequence.
        # We assume the original prompt batch had its own attention mask, but it
        # isn't passed in; we conservatively use 1's for the prompt section
        # (left-padding has already been handled at sampling time).
        B, T_full = prompts_responses.size()
        # Build attention from pad_mask (response side) + all-ones on prompt side.
        prompt_attn = torch.ones((B, prompt_len), dtype=torch.long, device=prompts_responses.device)
        attn = torch.cat([prompt_attn, (~pad_mask).long()], dim=1)

        with torch.no_grad():
            lora_to_base(self.model)
            outputs = self.model(input_ids=prompts_responses, attention_mask=attn)
            base_to_lora(self.model)
            logits = outputs.logits[:, prompt_len - 1 : -1]
            log_prob = F.log_softmax(logits, dim=-1)
            labels = prompts_responses[:, prompt_len:]
            per_token = torch.gather(log_prob, -1, labels.unsqueeze(2)).squeeze(2)
            per_token = torch.where(pad_mask, torch.zeros_like(per_token), per_token)
            lm_logreward = aggregate_lm_logreward(
                per_token,
                pad_mask,
                aggregation=self.cfg.reward.lm_aggregation,
                bottom_k=self.cfg.reward.lm_bottom_k,
            )
        return lm_logreward, pad_mask

    def _victim_and_classify(self, decoded_responses: List[str]) -> torch.Tensor:
        """Run the victim model on each attack prompt, then score with the classifier.

        Returns a ``[B]`` tensor of average classifier log-rewards (averaging
        over ``num_response_samples`` victim responses per attack).
        """
        victim_prompts = [_make_chat_prompt(self.victim_tokenizer, x) for x in decoded_responses]

        # vLLM lives on victim_device; temporarily switch CUDA default device.
        prev = torch.cuda.current_device()
        torch.cuda.set_device(self.cfg.gpu.victim_device)
        try:
            if self.cfg.model.victim_lora_path:
                llm_outputs = self.victim_model.generate(
                    victim_prompts,
                    self.sampling_params,
                    use_tqdm=False,
                    lora_request=LoRARequest("victim_adapter", 1, self.cfg.model.victim_lora_path),
                )
            else:
                llm_outputs = self.victim_model.generate(
                    victim_prompts, self.sampling_params, use_tqdm=False
                )
        finally:
            torch.cuda.set_device(prev)

        flat_prompts: List[str] = []
        flat_responses: List[str] = []
        for i, out in enumerate(llm_outputs):
            for choice in out.outputs:
                flat_responses.append(choice.text)
                flat_prompts.append(decoded_responses[i])

        c_scores = torch.tensor(
            self.toxicity_fn.compute(flat_prompts, flat_responses),
            device=self.device,
            dtype=torch.float,
        )
        # Average over multiple victim samples per attack.
        chunks = torch.split(c_scores, self.cfg.model.num_response_samples, dim=0)
        return torch.stack(chunks, dim=0).mean(1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _online_sample(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Generate on-policy attacks, score them, and add to the buffer."""
        cfg = self.cfg
        if random.random() < cfg.gen.temp_mix_prob:
            temperature = random.uniform(cfg.gen.temp_low, cfg.gen.temp_high)
        else:
            temperature = cfg.gen.sample_temperature
        out = generate_with_logprobs(
            self.model,
            prompt_ids=batch["input_ids"],
            prompt_attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            max_len=cfg.gen.max_len,
            logpf_aggregation=cfg.gen.logpf_aggregation,
        )
        prompts_responses: torch.Tensor = out["actions"]
        prompt_len = batch["input_ids"].size(1)
        responses = prompts_responses[:, prompt_len:]

        lm_logreward, _ = self._lm_logreward_from_reference(prompts_responses, prompt_len)
        if cfg.reward.disable_lm_reward:
            lm_logreward = torch.zeros_like(lm_logreward)

        decoded = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        c_log_reward = self._victim_and_classify(decoded)

        # Add to buffer using stored (raw) rewards.
        beta = cfg.reward.beta
        log_reward = lm_logreward + (c_log_reward / beta)
        embs = self.sentence_encoder.encode(decoded, convert_to_tensor=True)
        self.rbuffer.add_batch(
            responses,
            decoded,
            embs,
            c_log_reward,
            lm_logreward,
            log_reward,
            prompt_ids=batch["input_ids"],
        )

        return {
            "sum_logpf": out["sum_logpf"],
            "log_z": out["log_z"],
            "lm_log_reward": lm_logreward,
            "c_log_reward": c_log_reward,
            "decoded_responses": decoded,
            "is_on_policy": True,
        }

    def _offline_sample(self, n: int) -> Dict:
        """Sample ``n`` items from the replay buffer and recompute log PF / log Z."""
        prompt_batch, response_batch, reward_batch = self.rbuffer.sample(n)
        if prompt_batch is None:
            raise RuntimeError("Buffer has no stored prompts; refresh the buffer.")

        prompt_batch = {k: v.to(self.device) for k, v in prompt_batch.items()}
        response_batch = {k: v.to(self.device) for k, v in response_batch.items()}
        reward_batch = {k: v.to(self.device) for k, v in reward_batch.items()}

        sum_logpf, log_z = self._logpf_and_logz(prompt_batch, response_batch)
        decoded = self.tokenizer.batch_decode(
            response_batch["input_ids"], skip_special_tokens=True
        )

        return {
            "sum_logpf": sum_logpf,
            "log_z": log_z,
            "lm_log_reward": reward_batch["lm_log_reward"],
            "c_log_reward": reward_batch["c_log_reward"],
            "decoded_responses": decoded,
            "is_on_policy": False,
        }

    def _logpf_and_logz(
        self, prompt_batch: Dict[str, torch.Tensor], response_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute log PF and log Z for buffer-sampled trajectories."""
        prompt_len = prompt_batch["input_ids"].size(1)
        concat = {
            k: torch.cat([prompt_batch[k], response_batch[k]], 1)
            for k in prompt_batch
        }
        outputs = self.model(**concat, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1][:, :prompt_len]
        pooled = _avg_pool(prompt_hidden, prompt_batch["attention_mask"])
        log_z = self.model.proj_z(pooled).squeeze(-1)

        logits = outputs.logits[:, prompt_len - 1 : -1]
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = torch.gather(
            log_prob, -1, response_batch["input_ids"].unsqueeze(-1)
        ).squeeze(-1)
        log_prob = log_prob.masked_fill(response_batch["attention_mask"] == 0, 0.0)

        if self.cfg.gen.logpf_aggregation == "mean":
            lens = torch.sum(response_batch["attention_mask"].float(), dim=1)
            sum_logpf = torch.sum(log_prob, dim=1) / torch.clamp(lens, min=1.0)
        else:
            sum_logpf = torch.sum(log_prob, dim=1)
        return sum_logpf, log_z

    def _paired_sample(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Combine on-policy and off-policy samples in a single batch."""
        cfg = self.cfg
        B = batch["input_ids"].size(0)
        n_on = max(1, int(B * cfg.buffer.paired_on_policy_ratio))
        n_off = B - n_on

        on_batch = {k: v[:n_on] for k, v in batch.items()}
        on = self._online_sample(on_batch)

        actual_off = min(n_off, self.rbuffer.size())
        if actual_off == 0:
            on["is_on_policy"] = [True] * n_on
            return on

        off = self._offline_sample(actual_off)

        return {
            "sum_logpf": torch.cat([on["sum_logpf"], off["sum_logpf"]], 0),
            "log_z": torch.cat([on["log_z"], off["log_z"]], 0),
            "lm_log_reward": torch.cat([on["lm_log_reward"], off["lm_log_reward"]], 0),
            "c_log_reward": torch.cat([on["c_log_reward"], off["c_log_reward"]], 0),
            "decoded_responses": on["decoded_responses"] + off["decoded_responses"],
            "is_on_policy": [True] * n_on + [False] * actual_off,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_log_reward(
        self, sample: Dict, step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temperature scheduling + LM clip, return ``(log_reward, final_c)``."""
        cfg = self.cfg
        gamma = self.lm_temp(step)
        beta = cfg.reward.beta

        final_lm = sample["lm_log_reward"] / gamma
        final_lm = self.lm_clip(final_lm)
        final_c = sample["c_log_reward"] / beta

        log_reward = (final_lm + final_c) * self.reward_temp(step)
        return log_reward, final_c

    def _compute_loss(
        self, sample: Dict, step: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        log_reward, final_c = self._compute_log_reward(sample, step)

        metrics: Dict[str, float] = {}

        if cfg.loss.kind == "ctb":
            # CTB does not need log_z; detach for logging only.
            sample["log_z"] = sample["log_z"].detach()
            losses, stats = ctb_loss(
                sample["sum_logpf"],
                log_reward,
                mode=cfg.loss.baseline_mode,
                sigma=self.ngp(step),
                pair_weight_temp=cfg.loss.pair_weight_temp,
                c_log_reward=final_c,
                min_pairwise_c_reward=None,
                return_stats=cfg.debug,
            )
            if cfg.debug:
                metrics["graph/is_connected"] = stats.is_connected
                metrics["graph/num_components"] = stats.num_components
                metrics["graph/masked_pair_ratio"] = stats.masked_pair_ratio
        elif cfg.loss.kind == "tb":
            losses = tb_loss(
                sample["log_z"],
                sample["sum_logpf"],
                log_reward,
                log_z_bias=cfg.loss.log_z_bias,
            )
        else:
            raise ValueError(f"Unknown loss kind: {cfg.loss.kind!r}")

        loss = losses.mean()
        metrics.update({
            "loss/train": loss.item(),
            "log_z/train": float(sample["log_z"].detach().mean().item()),
            "sum_logpf/train": float(sample["sum_logpf"].detach().mean().item()),
            "c_log_reward/train": float(sample["c_log_reward"].mean().item()),
            "lm_log_reward/train": float(sample["lm_log_reward"].mean().item()),
            "log_reward/train": float(log_reward.detach().mean().item()),
            "ngp_sigma": self.ngp(step),
            "lm_temp_gamma": self.lm_temp(step),
        })
        return loss, metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, output_dir: str, step: int) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(
            {
                "global_step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "proj_z": self.model.proj_z.state_dict(),
            },
            os.path.join(output_dir, "ckpt.pt"),
        )
        self.rbuffer.save(os.path.join(output_dir, "buffer.pkl.gz"))

    def _maybe_resume(self) -> int:
        if not self.cfg.io.resume:
            return 1
        root = os.path.join(self.cfg.io.save_dir, self.cfg.io.exp_name)
        if not os.path.exists(root):
            return 1
        steps = sorted(
            [int(x) for x in os.listdir(root) if x.isdigit()], reverse=True
        )
        if not steps:
            return 1
        ckpt_dir = os.path.join(root, str(steps[0]))
        print(f"Resuming from {ckpt_dir}")
        _base = AutoModelForCausalLM.from_pretrained(self.cfg.model.sft_ckpt)
        _peft = PeftModel.from_pretrained(_base, ckpt_dir)
        self.model.load_state_dict(_peft.state_dict(), strict=False)

        ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location="cpu")
        self.model.proj_z.load_state_dict(ckpt["proj_z"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        buf_path = os.path.join(ckpt_dir, "buffer.pkl.gz")
        if os.path.exists(buf_path):
            self.rbuffer.load(buf_path)
        return ckpt["global_step"] + 1

    # ------------------------------------------------------------------
    # Buffer initialization
    # ------------------------------------------------------------------
    def _init_buffer_from_sft(self, batch: Dict[str, torch.Tensor]) -> None:
        """Seed the buffer with the SFT toxic-instructions before any training."""
        import json

        with open(self.cfg.io.few_shot_file, "r") as f:
            data = json.load(f)
        instructions = [" " + x["instruction"].strip() for x in data]
        cap = self.cfg.buffer.init_max_sft_samples
        if cap is not None and len(instructions) > cap:
            instructions = instructions[:cap]

        prev_pad = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        tok = self.tokenizer(
            instructions, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        self.tokenizer.padding_side = prev_pad

        responses = tok["input_ids"]
        eos = torch.full((responses.size(0), 1), self.tokenizer.eos_token_id, dtype=torch.long)
        responses = torch.cat([responses, eos], 1)

        loader = DataLoader(TensorDataset(responses), self.cfg.optim.batch_size, shuffle=False)
        for (chunk,) in tqdm(loader, desc="init buffer", leave=False):
            chunk = chunk.to(batch["input_ids"].device)
            prompts = batch["input_ids"].repeat(chunk.size(0), 1)
            attention = batch["attention_mask"].repeat(chunk.size(0), 1)
            prompts_responses = torch.cat([prompts, chunk], dim=1)

            lm_logreward, _ = self._lm_logreward_from_reference(
                prompts_responses, prompts.size(1)
            )
            if self.cfg.reward.disable_lm_reward:
                lm_logreward = torch.zeros_like(lm_logreward)

            decoded = self.tokenizer.batch_decode(chunk, skip_special_tokens=True)
            c_log_reward = self._victim_and_classify(decoded)

            final_lm = self.lm_clip(lm_logreward)
            log_reward = final_lm + (c_log_reward / self.cfg.reward.beta)
            embs = self.sentence_encoder.encode(decoded, convert_to_tensor=True)
            self.rbuffer.add_batch(
                chunk, decoded, embs, c_log_reward, lm_logreward, log_reward,
                prompt_ids=prompts,
            )

    def _init_buffer_from_base_policy(self, meta_batch: Dict[str, torch.Tensor]) -> None:
        """Generate extra trajectories from the *base* (LoRA-disabled) model to seed the buffer.

        With strict toxicity classifiers (e.g. Llama-Guard-3) and a high
        ``buffer.min_c_reward`` threshold, the SFT-only init in
        :meth:`_init_buffer_from_sft` often leaves the buffer empty (every
        SFT instruction scores too low to enter). Sampling several batches
        from the base policy gives the buffer a chance to find at least a
        few seed attacks that pass the threshold, which is critical for
        off-policy training to ever get going.

        ``cfg.buffer.init_extra_samples`` controls how many batches we draw.
        """
        cfg = self.cfg
        n_extra = cfg.buffer.init_extra_samples
        if n_extra <= 0:
            return

        for _ in tqdm(range(n_extra), desc="init buffer (base policy)", leave=False):
            with torch.no_grad():
                prompts_responses = self.model.generate(
                    **meta_batch,
                    do_sample=True,
                    max_new_tokens=cfg.gen.max_len,
                    temperature=0.7,
                    top_p=0.9,
                    min_new_tokens=cfg.gen.min_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            prompt_len = meta_batch["input_ids"].size(1)
            responses = prompts_responses[:, prompt_len:]

            lm_logreward, _ = self._lm_logreward_from_reference(
                prompts_responses, prompt_len
            )
            if cfg.reward.disable_lm_reward:
                lm_logreward = torch.zeros_like(lm_logreward)

            decoded = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            c_log_reward = self._victim_and_classify(decoded)

            final_lm = self.lm_clip(lm_logreward)
            log_reward = final_lm + (c_log_reward / cfg.reward.beta)
            embs = self.sentence_encoder.encode(decoded, convert_to_tensor=True)
            self.rbuffer.add_batch(
                responses, decoded, embs, c_log_reward, lm_logreward, log_reward,
                prompt_ids=meta_batch["input_ids"],
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg

        # Load or initialize buffer
        if cfg.buffer.buffer_path and os.path.exists(cfg.buffer.buffer_path):
            print(f"Loading pre-initialized buffer from {cfg.buffer.buffer_path}")
            self.rbuffer.load(cfg.buffer.buffer_path)
        elif self.rbuffer.size() == 0:
            print("Initializing buffer from scratch...")
            lora_to_base(self.model)
            batch = next(self.train_iter).to(self.device)
            self._init_buffer_from_sft(batch)
            # Broadcast the meta-prompt to batch_size copies and draw a few
            # extra base-policy samples so the buffer has some seed attacks.
            broadcast = {k: v.repeat(cfg.optim.batch_size, 1) for k, v in batch.items()}
            self._init_buffer_from_base_policy(broadcast)
            base_to_lora(self.model)
            print(f"Buffer seeded: size={self.rbuffer.size()}, "
                  f"stats={self.rbuffer.get_stats()}")

        # Prepare a fixed broadcast of the meta-prompt for each step.
        meta_batch = next(self.train_iter).to(self.device)
        meta_batch = {k: v.repeat(cfg.optim.batch_size, 1) for k, v in meta_batch.items()}

        t = tqdm(
            range(self.start_step, cfg.optim.train_steps + 1),
            desc="train",
            dynamic_ncols=True,
        )
        for global_step in t:
            self.model.train()
            self.optimizer.zero_grad()
            step_metrics: Dict[str, List[float]] = defaultdict(list)

            for _ in range(cfg.optim.grad_acc_steps):
                if cfg.buffer.paired_sampling:
                    sample = self._paired_sample(meta_batch)
                else:
                    if random.random() < cfg.buffer.buffer_sample_prob and self.rbuffer.size() > 0:
                        sample = self._offline_sample(cfg.optim.batch_size)
                    else:
                        sample = self._online_sample(meta_batch)
                loss, metrics = self._compute_loss(sample, global_step)
                (loss / cfg.optim.grad_acc_steps).backward()
                for k, v in metrics.items():
                    step_metrics[k].append(v)

            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            avg = {k: float(np.mean(vs)) for k, vs in step_metrics.items()}
            wandb.log(avg, step=global_step)
            display = {k: v for k, v in avg.items() if "loss" in k or "log_reward" in k}
            t.set_description(f"step {global_step}: {formatted_dict(display)}")

            if global_step % cfg.io.save_interval == 0:
                ckpt_dir = os.path.join(
                    cfg.io.save_dir, cfg.io.exp_name, str(global_step)
                )
                self._save(ckpt_dir, global_step)

        final_dir = os.path.join(cfg.io.save_dir, cfg.io.exp_name, "latest")
        self._save(final_dir, cfg.optim.train_steps)
        wandb.finish()
