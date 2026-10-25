#!/usr/bin/env python
"""Check DLite model loading, generation, and metric accounting on one GPU.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_dlite_backend.py \
        --target /models/Qwen3-4B \
        --draft /models/dlite_qwen3_4b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentspec.backends.deepspec_backend import DeepSpecBackend
from agentspec.config import BackendConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="Target model path")
    parser.add_argument("--draft", required=True, help="DLite draft checkpoint path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--turns", type=int, default=2,
                        help="conversation turns; each turn re-prefills the whole history")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BackendConfig(
        backend="spec",
        target_name_or_path=args.target,
        draft_name_or_path=args.draft,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[smoke] target={args.target}\n[smoke] draft={args.draft}")
    backend = DeepSpecBackend(cfg, local_rank=args.local_rank)
    block = backend.evaluator.draft_model.block_size
    print(f"[smoke] loaded: block_size={block} "
          f"(proposal={block - 1} draft tokens/round) "
          f"stop_tokens={sorted(backend._stop_token_ids or [])}")

    messages = [{"role": "user", "content": "Write a bash command that lists the five "
                                           "largest files in the current directory."}]
    failures = 0
    try:
        for turn in range(1, args.turns + 1):
            prompt_tokens = backend.count_tokens(messages)
            result = backend.generate(messages, max_new_tokens=args.max_new_tokens)
            accept_len = (
                result.accepted_sum / result.verify_count if result.verify_count else 0.0
            )
            tok_s = (
                result.num_output_tokens / result.gen_time_s
                if result.gen_time_s > 0 else 0.0
            )
            print(f"\n[smoke] turn {turn}: ctx={prompt_tokens}tok in={result.num_input_tokens} "
                  f"out={result.num_output_tokens} verify_rounds={result.verify_count} "
                  f"accept_len={accept_len:.2f} proposal={result.proposal_sum} "
                  f"time={result.gen_time_s:.2f}s tok/s={tok_s:.1f}")
            print(f"[smoke]   text: {result.text[:180]!r}")
            # Any turn that produced no text or no verification round is broken.
            if not result.text.strip() or result.verify_count == 0:
                print("[smoke]   FAIL: empty output or no verification round")
                failures += 1
            if result.num_input_tokens != prompt_tokens:
                print(f"[smoke]   FAIL: count_tokens ({prompt_tokens}) != "
                      f"num_input_tokens ({result.num_input_tokens})")
                failures += 1
            # Accept length must stay within [1, proposal+1] per round.
            max_accept = result.accepted_sum / result.verify_count if result.verify_count else 0.0
            if not 1.0 <= max_accept <= block:
                print(f"[smoke]   FAIL: accept_len {max_accept} outside [1, {block}]")
                failures += 1
            # Grow the conversation the way an agent episode does.
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "user", "content": "Now show the same for /var/log, "
                                                        "and explain each flag you used."})
    finally:
        backend.clean_up()

    print("\n[smoke] OK" if failures == 0 else f"\n[smoke] {failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
