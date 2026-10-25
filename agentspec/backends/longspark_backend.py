"""LongSpark speculative-decoding backend built on the vendored runtime.

LongSpark does not run inside transformers: the target lives in vendored SGLang
model runners that occupy their own GPU ranks (spawned subprocesses), while the
fixed-cost drafter runs in this process on one extra GPU.  One agent turn is
therefore driven exactly like LongSpark's closed `run_finite` loop with
concurrency 1:

    prefill (returns the first token) ->
        repeat: drafter proposes 7 tokens -> target verifies the block ->
                drafter folds the accepted KV back into its bounded state
    -> release the request slot

Because the target verifies every proposal, the emitted distribution is the
target's own; only the number of target forward passes changes.

Metric mapping (kept identical to `DeepSpecBackend` so runs are comparable):
  * `verify_count`   one entry per verification round.
  * `accepted_sum`   sum of `accepted + 1` per round (accepted draft tokens plus
                     the bonus token the target always emits), i.e. the same
                     `acceptance_lengths` definition the DeepSpec loop uses.
  * `proposal_sum`   sum of effective proposal lengths = `block_size` per round.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from ..config import BackendConfig
from .base import BaseBackend, GenerationResult

# agentspec/backends/ -> agentspec/ -> <project root>
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor"

_BOOTSTRAPPED = False


def _bootstrap_vendor_path() -> None:
    """Put the vendored SGLang / SpecForge source trees on `sys.path`.

    Upstream imports them from the source tree instead of installing them, so
    this must happen before anything imports `sglang` or `specforge`.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    sglang_python = VENDOR_ROOT / "sglang" / "python"
    if not sglang_python.is_dir():
        raise FileNotFoundError(
            "LongSpark requires agentspec/vendor/sglang. Extract "
            "agentspec/vendor/sglang.tar.gz into agentspec/vendor/ before "
            "running the LongSpark backend."
        )
    from ..vendor.longspark.paths import VENDOR_PATHS

    for path in (str(sglang_python), str(VENDOR_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # Keep the vendored runtime importable from spawned target ranks too.
    existing = os.environ.get("PYTHONPATH", "")
    missing = [str(p) for p in VENDOR_PATHS if str(p) not in existing]
    if missing:
        os.environ["PYTHONPATH"] = os.pathsep.join([*missing, existing]) if existing else os.pathsep.join(missing)
    _BOOTSTRAPPED = True


def _install_kernel_flags(target_path: str) -> None:
    """Kernel switches the vendored SGLang reads at import/run time.

    Must run before `sglang` is imported. `LONGSPARK_GQA5` additionally selects
    the GQA-5 attention adapter, which Qwen3-14B (40 query / 8 KV heads) needs.
    """
    from ..vendor.longspark.gpus import KERNEL_FLAGS

    for key, value in KERNEL_FLAGS.items():
        os.environ.setdefault(key, value)

    config = json.loads((Path(target_path) / "config.json").read_text())
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    os.environ["LONGSPARK_GQA5"] = "1" if heads // kv_heads == 5 else "0"


def _build_prefill_client():
    """Subclass the vendored client so one turn costs a single prefill RPC.

    Defined lazily: importing `longspark.engine.target` pulls in the vendored
    SGLang runtime, which must not happen when this module is merely imported.
    """

    from ..vendor.longspark.engine import target as tm

    class BatchedPrefillClient(tm.TargetClient):
        """Sends the whole prompt as one final prefill chunk.

        Same as upstream `longspark.worker.BatchedPrefillClient`: the base
        client chunks the prompt and drives the drafter through
        `prefill_consumer` itself, which is unnecessary for a single request.
        """

        def prefill(self, ids, blocks, temperature, *, preserved_anchors=None):
            if preserved_anchors is None:
                preserved_anchors = [None] * len(ids)
            self._send(dict(op="prefill", ids=ids, blocks=blocks, temperature=temperature,
                            preserved_anchors=preserved_anchors, final_chunk=True))
            if self.channel is None:
                return self._responses()[0]
            payload = self.channel.receive()
            self._responses()
            return payload

    return BatchedPrefillClient


class LongSparkBackend(BaseBackend):
    name = "longspark"

    def __init__(self, cfg: BackendConfig, local_rank: int = 0):
        if not cfg.draft_name_or_path:
            raise ValueError("LongSparkBackend requires draft_name_or_path")
        self.cfg = cfg
        self.local_rank = local_rank
        self._request_seq = 0

        _install_kernel_flags(cfg.target_name_or_path)
        _bootstrap_vendor_path()

        from ..vendor.longspark.engine import lifecycle
        from ..vendor.longspark.engine import target as tm
        from ..vendor.longspark.engine.bench_utils import warmup
        from ..vendor.longspark.engine.config import SplitConfig
        from ..vendor.longspark.engine.draft import LongSparkDraft
        from ..vendor.longspark.patches import install_target_patches
        from transformers import AutoTokenizer

        self._lifecycle = lifecycle
        self._tm = tm

        # Applies the registered-IPC / GQA-5 patches here and makes the spawned
        # target ranks apply them too (`spawn` re-imports the entry point).
        install_target_patches()

        target_tp = int(cfg.longspark_target_tp)
        if cfg.longspark_target_devices:
            target_devices = tuple(int(x) for x in cfg.longspark_target_devices.split(",") if x.strip())
        else:
            # Each worker owns target_tp + 1 consecutive GPUs.
            base = local_rank * (target_tp + 1)
            target_devices = tuple(range(base, base + target_tp))
        draft_device = (
            int(cfg.longspark_draft_device)
            if cfg.longspark_draft_device >= 0
            else local_rank * (target_tp + 1) + target_tp
        )
        if len(target_devices) != target_tp:
            raise ValueError(
                f"longspark_target_devices must list exactly target_tp={target_tp} devices, got {target_devices}"
            )
        if draft_device in target_devices:
            raise ValueError(f"draft device {draft_device} collides with target devices {target_devices}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_name_or_path)
        self.split_config = SplitConfig(
            target_tp=target_tp,
            target_devices=target_devices,
            draft_device=draft_device,
            target_kv_gib_per_rank=float(cfg.longspark_target_kv_gib),
            draft_state_gib=float(cfg.longspark_draft_state_gib),
            max_running_requests=1,          # one agent turn at a time
            cuda_graph=True,
            fuse_kv_export=False,
            position_variant=cfg.longspark_position_variant,
            context_length=int(cfg.longspark_context_length),
            prefill_chunk_size=int(cfg.longspark_prefill_chunk_size),
            eos_token_ids=self._resolve_eos(cfg),
        )
        self.target = _build_prefill_client()(
            self.split_config,
            model_path=cfg.target_name_or_path,
            method="longspark",
            draft_path=cfg.draft_name_or_path,
            seed=int(cfg.seed),
        )
        self.draft = LongSparkDraft(
            self.split_config,
            model_path=cfg.target_name_or_path,
            draft_path=cfg.draft_name_or_path,
            target_ready=self.target.ready,
        )
        self.target.prefill_consumer = lambda states: self.draft.update(states, prefill=True)

        self.block_size = int(self.draft.model.config.block_size)
        self._eos = set(self.split_config.eos_token_ids)
        self.device = self.draft.device

        # Capture the batch-1 CUDA graphs before the first real request.
        warm_ids = self.tokenizer("The following is a short warm-up sentence used to capture decode graphs.",
                                  add_special_tokens=False).input_ids
        warmup(self.target, self.draft, self.split_config, warm_ids, float(cfg.temperature))
        self.target.reset_measurement(int(cfg.seed))

    # ------------------------------------------------------------------ setup

    def _resolve_eos(self, cfg: BackendConfig) -> tuple[int, ...]:
        """Stop tokens for the target-side commit logic."""
        if cfg.longspark_eos_token_ids:
            return tuple(int(x) for x in cfg.longspark_eos_token_ids.split(",") if x.strip())
        ids: list[int] = []
        raw = getattr(self.tokenizer, "eos_token_id", None)
        for token_id in raw if isinstance(raw, (list, tuple)) else ([raw] if raw is not None else []):
            token_id = int(token_id)
            if token_id not in ids:
                ids.append(token_id)
        # Qwen3 chat models emit <|im_end|>; <|endoftext|> guards raw completions.
        for token_id in (151645, 151643):
            if token_id not in ids:
                ids.append(token_id)
        return tuple(ids)

    # --------------------------------------------------------------- encoding

    def _encode(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def count_tokens(self, messages: list[dict]) -> int:
        return int(self._encode(messages).shape[1])

    def _sync(self) -> None:
        """Wait for the drafter's device (no-op off CUDA)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------- generation

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        extra_stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        input_ids = self._encode(messages)[0].tolist()
        max_new_tokens = int(max_new_tokens)
        prompt_len = len(input_ids)
        limit = self.split_config.generation_context_limit
        if prompt_len + max_new_tokens > limit:
            raise ValueError(
                f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds the "
                f"LongSpark context budget ({limit}); raise --longspark-context-length "
                f"(with --longspark-position-variant fixed_ntk) or lower --max-context-tokens"
            )

        eos = set(self._eos)
        for token_id in extra_stop_token_ids or []:
            eos.add(int(token_id))

        rid = f"turn{self._request_seq}"
        self._request_seq += 1
        temperature = float(self.cfg.temperature)

        start = time.perf_counter()
        try:
            payload = self.target.prefill([rid], [input_ids], temperature, preserved_anchors=[None])
            self.draft.update(payload["states"], prefill=True)
            self._sync()
            anchor = int(payload["bonus"][0])
            generated = [anchor]
            rounds: list[tuple[int, int, int]] = []

            while len(generated) < max_new_tokens and anchor not in eos:
                proposed, logits, sampled_q = self.draft.propose([rid], [anchor], temperature)
                self._sync()
                proposals = proposed.tolist()[0]
                blocks = [[anchor] + proposals]
                payload = self.target.verify(
                    [rid], blocks, temperature, draft_logits=logits, sampled_q=sampled_q
                )
                self.draft.update(payload["states"])
                accepted = int(payload["commit_lengths"][0]) - 1
                bonus = int(payload["bonus"][0])
                emitted = accepted_emitted = 0
                for position, token in enumerate(proposals[:accepted] + [bonus]):
                    if len(generated) >= max_new_tokens:
                        break
                    generated.append(int(token))
                    emitted += 1
                    accepted_emitted += int(position < accepted)
                    if token in eos:
                        break
                # (accepted, accepted tokens that reached the output, tokens emitted)
                rounds.append((accepted, accepted_emitted, emitted))
                anchor = bonus
                if generated[-1] in eos:
                    break
        finally:
            # Always hand the slot back, otherwise the bounded drafter state
            # pool leaks one slot per failed turn.
            try:
                self.target.release([rid])
                self.draft.release([rid])
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        gen_time_s = time.perf_counter() - start

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            num_input_tokens=prompt_len,
            num_output_tokens=len(generated),
            verify_count=len(rounds),
            accepted_sum=sum(accepted + 1 for accepted, _, _ in rounds),
            proposal_sum=self.block_size * len(rounds),
            gen_time_s=gen_time_s,
        )

    def clean_up(self) -> None:
        """Reverse construction order: drafter graphs, target ranks, TP group."""
        try:
            self._lifecycle.release_draft_graphs(self.draft)
        finally:
            try:
                self.target.close(graceful=True)
            finally:
                if torch.distributed.is_initialized():
                    self._lifecycle.cleanup_parallel_runtime()



