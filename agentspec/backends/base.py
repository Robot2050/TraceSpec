"""Backend interface: one `generate` call = one agent turn.

A backend wraps a (target, draft) model pair and exposes a single
chat-message-level generation API. The DeepSpec backend performs true
speculative decoding; the AR backend is the target-only baseline used to
quantify the end-to-end speedup; the mock backend is a no-GPU smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    text: str
    num_input_tokens: int
    num_output_tokens: int
    verify_count: int        # verification rounds (AR: == num_output_tokens)
    accepted_sum: int        # sum of per-round acceptance lengths (AR: == num_output_tokens)
    proposal_sum: int        # sum of per-round effective proposal lengths (AR: 0)
    gen_time_s: float        # wall time, prefill + decode


class BaseBackend:
    name = "base"

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        extra_stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def count_tokens(self, messages: list[dict]) -> int:
        """Token count of the full prompt for these messages (with generation
        prompt), used by the context guard."""
        raise NotImplementedError

    def clean_up(self) -> None:
        pass
