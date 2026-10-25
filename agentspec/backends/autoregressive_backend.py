"""Target-only autoregressive baseline backend.

Same model, same chat template, same per-turn stateless generation as the
speculative backend, but without a draft model. Used to quantify the
end-to-end speedup speculative decoding gives on full agent episodes.
acceptance_length is 1.0 by construction (one token per forward).
"""
from __future__ import annotations

import time

from ..config import BackendConfig
from .base import BaseBackend, GenerationResult


class AutoRegressiveBackend(BaseBackend):
    name = "ar"

    def __init__(self, cfg: BackendConfig, local_rank: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device=self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_name_or_path)

    def _encode(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

    def count_tokens(self, messages: list[dict]) -> int:
        return int(self._encode(messages).shape[1])

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        extra_stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        import torch

        input_ids = self._encode(messages)
        do_sample = float(self.cfg.temperature) >= 1e-5
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(self.cfg.temperature) if do_sample else None,
            eos_token_id=(self.model.generation_config.eos_token_id),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids, **gen_kwargs)
        gen_time_s = time.perf_counter() - start

        new_ids = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        num_output = int(new_ids.shape[0])
        return GenerationResult(
            text=text,
            num_input_tokens=int(input_ids.shape[1]),
            num_output_tokens=num_output,
            verify_count=num_output,      # one forward per token
            accepted_sum=num_output,      # acceptance length == 1 per round
            proposal_sum=0,
            gen_time_s=gen_time_s,
        )
