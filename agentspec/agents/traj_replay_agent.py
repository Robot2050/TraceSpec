"""Replay one recorded model call for each TraceSpec-Traj state.

Replay shares metric records and aggregation with dynamic evaluation, but does
not execute tools or trim the saved prefix. The phase metadata maps to the mode
field, and the tokenized input determines the context bucket. Tool-round counts
map to episode progress as follows: plan to step -1, act and answer to R, and
reflect to R - 1, where R is the number of preceding tool observations.

States above the configured context limit are recorded as context_limit. The
reference continuation remains in the dataset and is never shown to the model.
"""
from __future__ import annotations

import json
import logging
import time

from ..backends.base import BaseBackend
from ..config import (
    AgentConfig,
    MODE_ACT,
    MODE_ANSWER,
    MODE_PLAN,
    MODE_REFLECT,
)
from ..datasets import AgentTask
from ..metrics import EpisodeRecord, EpisodeRecorder
from .base import Action, LoopAgent

_logger = logging.getLogger(__name__)


class TrajReplayAgent(LoopAgent):
    """Implement the shared agent interface for fixed-prefix replay."""

    # ---- LoopAgent abstract interface: unused by the replay path ------------

    def initial_messages(self, task: AgentTask) -> list[dict]:
        return list(task.meta["input"])

    def parse_action(self, text: str) -> Action:  # pragma: no cover - unused
        return Action(kind="invalid", payload=text)

    def execute(self, action: Action) -> str:  # pragma: no cover - unused
        return ""

    # ---- replay episode ------------------------------------------------------

    @staticmethod
    def _step_for(phase: str, num_tool_rounds: int) -> int:
        """Map the cut position to the live agents' step convention (see the
        module docstring table)."""
        if phase == MODE_PLAN:
            return -1
        if phase == MODE_REFLECT:
            return max(0, num_tool_rounds - 1)
        # act and answer both record the number of completed tool rounds
        return num_tool_rounds

    def run(self, task: AgentTask, episode_id: str) -> EpisodeRecord:
        recorder = EpisodeRecorder(episode_id, task.dataset, task.task_id)
        meta = task.meta
        phase = meta.get("phase", MODE_ACT)
        num_tool_rounds = int(meta.get("num_tool_rounds", 0))
        messages = list(meta["input"])
        step = self._step_for(phase, num_tool_rounds)
        try:
            # Same context guard value as the live loop; nothing is trimmed —
            # over-long samples are recorded and skipped instead.
            if self._over_context_limit_for_replay(messages):
                _logger.warning(
                    f"[episode {episode_id}] context limit exceeded "
                    f"(bucket={meta.get('length_bucket')}); skipping generation"
                )
                recorder.episode.num_steps = num_tool_rounds
                return recorder.finish(
                    finish_reason="context_limit",
                    output=json.dumps(meta.get("output", {}), ensure_ascii=False)[:2000],
                )

            _logger.info(
                f"[episode {episode_id}] replaying phase={phase} "
                f"bucket={meta.get('length_bucket')} mode={meta.get('mode')} "
                f"rounds={num_tool_rounds}"
            )
            t0 = time.perf_counter()
            result = self.backend.generate(
                messages, max_new_tokens=self.cfg.max_new_tokens
            )
            elapsed = time.perf_counter() - t0
            _logger.info(
                f"[episode {episode_id}] done: ctx={result.num_input_tokens} "
                f"out={result.num_output_tokens} verify={result.verify_count} "
                f"accepted={result.accepted_sum} time={elapsed:.2f}s"
            )
            recorder.add_generation(
                step=step,
                mode=phase,
                num_input_tokens=result.num_input_tokens,
                num_output_tokens=result.num_output_tokens,
                verify_count=result.verify_count,
                accepted_sum=result.accepted_sum,
                proposal_sum=result.proposal_sum,
                gen_time_s=result.gen_time_s,
                backend=self.backend.name,
            )
            recorder.episode.num_steps = num_tool_rounds
            # `output` keeps the GENERATED text (same as live agents); the
            # reference continuation stays available through task.meta / the
            # dataset jsonl for offline fidelity diffs.
            return recorder.finish(finish_reason="replayed", output=result.text)
        except Exception as exc:  # keep one broken sample from killing the shard
            _logger.exception(f"[episode {episode_id}] exception in replay")
            return recorder.finish(finish_reason="error", output=f"{type(exc).__name__}: {exc}")

    def _over_context_limit_for_replay(self, messages: list[dict]) -> bool:
        try:
            ctx = self.backend.count_tokens(messages)
        except Exception:
            return False  # mock backends / tokenizers without templates
        return ctx > self.cfg.max_context_tokens
