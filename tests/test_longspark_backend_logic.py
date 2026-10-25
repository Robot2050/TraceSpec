"""Offline tests for the LongSpark turn loop and metric accounting.

Run with:
    python tests/test_longspark_backend_logic.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentspec.backends.longspark_backend import LongSparkBackend
from agentspec.config import BackendConfig

IM_END = 151645
END_OF_TEXT = 151643


class TokenizerOutput(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    eos_token_id = IM_END

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, **kwargs):
        ids = []
        for word in text.split():
            if word == "<|im_end|>":
                ids.append(IM_END)
            elif word == "<|endoftext|>":
                ids.append(END_OF_TEXT)
            else:
                ids.append(100 + len(ids) % 100)
        ids = ids or [100]
        return TokenizerOutput(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        words = []
        for token_id in ids:
            if skip_special_tokens and token_id in {IM_END, END_OF_TEXT}:
                continue
            if token_id == IM_END:
                words.append("<|im_end|>")
            elif token_id == END_OF_TEXT:
                words.append("<|endoftext|>")
            else:
                words.append(f"token_{token_id}")
        return " ".join(words)


class FakeTarget:
    """Returns a scripted `(accepted, bonus)` per verification round."""

    def __init__(self, script, prefill_bonus):
        self.script = script
        self.prefill_bonus = prefill_bonus
        self.round = 0
        self.released: list[str] = []
        self.states_seen = 0

    def prefill(self, ids, blocks, temperature, *, preserved_anchors=None):
        self.states_seen += 1
        return dict(states=["prefill-state"], bonus=[self.prefill_bonus],
                    commit_lengths=[len(blocks[0])])

    def verify(self, ids, blocks, temperature, *, draft_logits=None, sampled_q=None):
        if self.round >= len(self.script):
            raise AssertionError("verify called more times than scripted")
        accepted, bonus = self.script[self.round]
        self.round += 1
        return dict(states=[f"state-{self.round}"], bonus=[bonus],
                    commit_lengths=[accepted + 1])

    def release(self, ids):
        self.released.extend(ids)


class FakeDraft:
    """Proposes `block_size` distinguishable tokens per round."""

    def __init__(self, block_size=7):
        self.block_size = block_size
        self.device = torch.device("cpu")
        self.released: list[str] = []
        self.updates = 0

    def propose(self, ids, anchors, temperature):
        base = 1000 + 10 * len(self.released) + self.updates
        tokens = [base + i for i in range(self.block_size)]
        return torch.tensor([tokens]), None, None

    def update(self, states, prefill=False):
        self.updates += 1

    def release(self, ids):
        self.released.extend(ids)


def make_backend(script, prefill_bonus, *, block_size=7, eos=(IM_END,)):
    backend = LongSparkBackend.__new__(LongSparkBackend)
    backend.cfg = BackendConfig(backend="longspark", temperature=0.0)
    backend.local_rank = 0
    backend._request_seq = 0
    backend.tokenizer = FakeTokenizer()
    backend.split_config = SimpleNamespace(eos_token_ids=tuple(eos), generation_context_limit=40960)
    backend.target = FakeTarget(script, prefill_bonus)
    backend.draft = FakeDraft(block_size)
    backend.block_size = block_size
    backend._eos = set(eos)
    backend.device = torch.device("cpu")
    return backend


MESSAGES = [{"role": "user", "content": "Write a short bash command that lists files."}]


def check(name, got, want):
    status = "ok " if got == want else "FAIL"
    print(f"  [{status}] {name}: got={got} want={want}")
    return got == want


def case_normal_eos():
    print("case 1: two rounds, EOS in the bonus token of round 2")
    backend = make_backend(script=[(3, 3001), (2, IM_END)], prefill_bonus=2000)
    result = backend.generate(MESSAGES, max_new_tokens=64)
    ok = True
    ok &= check("num_output_tokens", result.num_output_tokens, 1 + 4 + 3)
    ok &= check("verify_count", result.verify_count, 2)
    ok &= check("accepted_sum (Σ accepted+1)", result.accepted_sum, (3 + 1) + (2 + 1))
    ok &= check("proposal_sum (block_size × rounds)", result.proposal_sum, 7 * 2)
    ok &= check("acceptance_length", round(result.accepted_sum / result.verify_count, 4), 3.5)
    ok &= check("slot released on target", backend.target.released, ["turn0"])
    ok &= check("slot released on draft", backend.draft.released, ["turn0"])
    ok &= check("EOS stripped from text", IM_END not in backend.tokenizer(result.text)["input_ids"], True)
    return ok


def case_max_new_tokens():
    print("case 2: max_new_tokens truncates mid-block")
    backend = make_backend(script=[(6, 4001), (6, 4002)], prefill_bonus=2000)
    result = backend.generate(MESSAGES, max_new_tokens=3)
    ok = True
    ok &= check("num_output_tokens", result.num_output_tokens, 3)
    ok &= check("verify_count", result.verify_count, 1)
    ok &= check("accepted_sum", result.accepted_sum, 6 + 1)
    ok &= check("proposal_sum", result.proposal_sum, 7)
    return ok


def case_immediate_eos():
    print("case 3: EOS already in the prefill bonus token")
    backend = make_backend(script=[], prefill_bonus=IM_END)
    result = backend.generate(MESSAGES, max_new_tokens=64)
    ok = True
    ok &= check("num_output_tokens", result.num_output_tokens, 1)
    ok &= check("verify_count", result.verify_count, 0)
    ok &= check("accepted_sum", result.accepted_sum, 0)
    ok &= check("proposal_sum", result.proposal_sum, 0)
    ok &= check("text is empty", result.text.strip(), "")
    return ok


def case_context_budget():
    print("case 4: over-long prompt raises instead of corrupting the target")
    backend = make_backend(script=[], prefill_bonus=2000)
    try:
        backend.generate(MESSAGES, max_new_tokens=40959)
        print("  [FAIL] expected ValueError")
        return False
    except ValueError as exc:
        print(f"  [ok ] raised ValueError: {str(exc)[:70]}...")
        return True


def case_count_tokens():
    print("case 5: count_tokens matches the encoded prompt length")
    backend = make_backend(script=[(1, 5001), (1, 5002), (1, IM_END)], prefill_bonus=2000)
    n = backend.count_tokens(MESSAGES)
    result = backend.generate(MESSAGES, max_new_tokens=4)
    ok = check("count_tokens == num_input_tokens", n, result.num_input_tokens)
    ok &= check("prompt has tokens", n > 0, True)
    return ok


def main() -> int:
    cases = [case_normal_eos, case_max_new_tokens, case_immediate_eos,
             case_context_budget, case_count_tokens]
    results = [fn() for fn in cases]
    print()
    print(f"{sum(results)}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
