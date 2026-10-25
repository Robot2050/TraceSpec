"""Speculative-decoding backend built on the vendored evaluators.

Uses the vendored evaluator classes (agentspec.vendor.deepspec:
DSpark / EAGLE3, Qwen3 / Gemma4) for model loading, drafting and
verification, but drives them per agent turn instead of per single-turn
dataset sample. Each call re-prefills the full conversation (the evaluators
are stateless across requests), which is exactly the long-context regime
this harness measures.
"""
from __future__ import annotations

import os
import socket
import time
from types import SimpleNamespace

from ..config import BackendConfig
from .base import BaseBackend, GenerationResult


def pick_free_port() -> int:
    """An ephemeral TCP port free right now, for the torch.distributed
    rendezvous. The harness entry point picks this once in the parent before
    spawning workers, so all ranks of one run share it while concurrent runs
    on the same machine each get their own."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _ensure_dist_env() -> None:
    """The vendored init_dist reads these from the environment."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        # Last-resort fallback for direct (out-of-harness) backend use;
        # run_agent_eval.main pins one free port per run before spawning.
        os.environ["MASTER_PORT"] = str(pick_free_port())


class DeepSpecBackend(BaseBackend):
    name = "spec"

    def __init__(self, cfg: BackendConfig, local_rank: int = 0):
        if not cfg.draft_name_or_path:
            raise ValueError("DeepSpecBackend requires draft_name_or_path")
        _ensure_dist_env()

        from transformers import AutoConfig

        from ..vendor.deepspec.eval.base_evaluator import resolve_stop_token_ids
        from ..vendor.deepspec.eval.dspark import (
            Gemma4DSparkEvaluator,
            Qwen3DSparkEvaluator,
        )
        from ..vendor.deepspec.eval.eagle3 import (
            Gemma4Eagle3Evaluator,
            Qwen3Eagle3Evaluator,
        )
        from ..vendor.dlite.eval import Qwen3DLiteEvaluator

        # Dispatch on the draft config's architecture field.
        evaluator_classes = {
            "Qwen3DSparkModel": Qwen3DSparkEvaluator,
            "Gemma4DSparkModel": Gemma4DSparkEvaluator,
            "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
            "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
            "Eagle3DraftModel": Qwen3Eagle3Evaluator,
            "DLiteDraftModel": Qwen3DLiteEvaluator,
        }

        draft_config = AutoConfig.from_pretrained(cfg.draft_name_or_path)
        arch = draft_config.architectures[0]
        if arch not in evaluator_classes:
            raise ValueError(
                f"unsupported draft architecture {arch!r}; "
                f"expected one of {sorted(evaluator_classes)}"
            )

        args = SimpleNamespace(
            target_name_or_path=cfg.target_name_or_path,
            draft_name_or_path=cfg.draft_name_or_path,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            tree_draft_temperature=cfg.tree_draft_temperature,
            confidence_threshold=cfg.confidence_threshold,
            tensorboard_dir=None,
            step=None,
            seed=cfg.seed,
            tasks=[],
        )
        self.cfg = cfg
        self.evaluator = evaluator_classes[arch](local_rank, args)
        # The DSpark evaluator spins up a confidence-calibration recorder when
        # the draft has a confidence head; it only adds per-round overhead for
        # agent use, so disable it.
        if hasattr(self.evaluator, "confidence_head_recorder"):
            self.evaluator.confidence_head_recorder = None

        self.tokenizer = self.evaluator.tokenizer
        self.device = self.evaluator.device
        self._stop_token_ids = resolve_stop_token_ids(
            self.evaluator.target_model, self.tokenizer
        )

    def _encode(self, messages: list[dict]):
        from ..vendor.deepspec.data.parser import encode_chat_messages

        return encode_chat_messages(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.device)

    def count_tokens(self, messages: list[dict]) -> int:
        return int(self._encode(messages).shape[1])

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        extra_stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        input_ids = self._encode(messages)
        stop_token_ids = list(self._stop_token_ids or [])
        for token_id in extra_stop_token_ids or []:
            if token_id not in stop_token_ids:
                stop_token_ids.append(token_id)

        # generate_one_sample reads max_new_tokens from args; set per call.
        self.evaluator.args.max_new_tokens = int(max_new_tokens)
        start = time.perf_counter()
        response = self.evaluator.generate_one_sample(
            input_ids=input_ids,
            stop_token_ids=stop_token_ids or None,
        )
        gen_time_s = time.perf_counter() - start

        output_ids = response.output_ids[0][response.num_input_tokens:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            num_input_tokens=int(response.num_input_tokens),
            num_output_tokens=int(response.num_output_tokens),
            verify_count=int(response.verify_count),
            accepted_sum=int(sum(response.acceptance_lengths)),
            proposal_sum=int(sum(response.proposal_lengths)),
            gen_time_s=gen_time_s,
        )

    def clean_up(self) -> None:
        self.evaluator.clean_up()
