"""Deterministic no-GPU backend for pipeline smoke tests.

Produces syntactically valid agent responses (so the tool/parse/record
pipeline can be exercised end to end) and fabricates accepted-length metrics
as a deterministic function of context length, so bucketing and aggregation
can be validated without loading a model.
"""
from __future__ import annotations

import time

from ..config import BackendConfig, MODE_ANSWER, MODE_PLAN, MODE_REFLECT
from .base import BaseBackend, GenerationResult


class MockBackend(BaseBackend):
    name = "mock"

    def __init__(self, cfg: BackendConfig, local_rank: int = 0, script: str = "swe"):
        self.cfg = cfg
        self.script = script
        self._calls: list[int] = []

    def count_tokens(self, messages: list[dict]) -> int:
        # Crude but monotonic pseudo-token count: ~4 chars per token.
        return sum(len(m["content"]) for m in messages) // 4 + 32

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        extra_stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        del max_new_tokens, extra_stop_token_ids
        num_input = self.count_tokens(messages)
        self._calls.append(num_input)
        last = messages[-1]
        mode_hint = last["content"][:400].lower()

        if last["role"] == "user" and (
            "before taking any action" in mode_hint  # opening plan
            or "new plan" in mode_hint               # re-plan prompts
        ):
            text = "Plan: inspect the relevant code, reproduce the issue, patch it, then submit."
        elif last["role"] == "user" and (
            "reflect on" in mode_hint
            or "evaluate" in mode_hint
            or "decision line" in mode_hint          # DECISION_INSTRUCTION
            or "analyze honestly" in mode_hint       # re-plan deep analysis
        ):
            text = (
                "Reflection: progress so far is on track; continue narrowing "
                "down the root cause.\nDECISION: CONTINUE"
            )
        else:
            text = self._scripted_action(messages)

        num_output = max(1, len(text) // 4)
        # Fabricate a context-dependent acceptance length: decays as the
        # pseudo-context grows, deterministic per input size.
        verify_count = max(1, num_output // 4)
        accepted_len = max(1.2, 4.0 - num_input / 16384.0)
        accepted_sum = int(accepted_len * verify_count)
        time.sleep(0.001)
        return GenerationResult(
            text=text,
            num_input_tokens=num_input,
            num_output_tokens=num_output,
            verify_count=verify_count,
            accepted_sum=accepted_sum,
            proposal_sum=accepted_sum * 2,
            gen_time_s=0.001,
        )

    def _scripted_action(self, messages: list[dict]) -> str:
        n = len(self._calls)
        if self.script == "search":
            # Search long enough to exercise the streak machinery: the forced
            # reflect fires at streak 3 and the force-read at streak 5.
            if n <= 9:
                return "```search\nsample query\n```"
            return "```answer\nmock answer\n```"
        if self.script == "traj":
            # Replay smoke test: return a short plausible continuation. The
            # replay agent records whatever comes back, so any non-empty text
            # exercises the record path.
            return "Replayed continuation (mock): proceeding with the next step."
        # swe script: three failing commands (triggering the forced reflect at
        # failure streak 3), then a successful one, then submit
        if n <= 4:
            return "```bash\nfalse\n```"
        if n <= 6:
            return "```bash\nls\n```"
        return "```bash\nsubmit\n```"
