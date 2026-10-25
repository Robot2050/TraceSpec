#!/usr/bin/env python
"""Check LongSpark model loading, generation, and metric accounting.

The split runtime requires one target GPU and one draft GPU.

Example:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_longspark_backend.py \
        --target /models/Qwen3-4B \
        --draft /models/LongSpark-Qwen3-4B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentspec.backends.longspark_backend import LongSparkBackend
from agentspec.config import BackendConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="target model path")
    parser.add_argument("--draft", required=True, help="LongSpark drafter path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--turns", type=int, default=2,
                        help="conversation turns; each turn re-prefills the whole history")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="worker index; rank i uses GPUs (2i, 2i+1) by default")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BackendConfig(
        backend="longspark",
        target_name_or_path=args.target,
        draft_name_or_path=args.draft,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[smoke] target={args.target}\n[smoke] draft={args.draft}")
    backend = LongSparkBackend(cfg, local_rank=args.local_rank)
    print(f"[smoke] loaded: block_size={backend.block_size} eos={sorted(backend._eos)} "
          f"target_devices={backend.split_config.target_devices} "
          f"draft_device={backend.split_config.draft_device}")

    messages = [{"role": "user", "content": "Write a bash command that lists the five "
                                            "largest files in the current directory."}]
    failures = 0
    try:
        for turn in range(1, args.turns + 1):
            prompt_tokens = backend.count_tokens(messages)
            result = backend.generate(messages, max_new_tokens=args.max_new_tokens)
            accept_len = result.accepted_sum / result.verify_count if result.verify_count else 0.0
            tok_s = result.num_output_tokens / result.gen_time_s if result.gen_time_s > 0 else 0.0
            print(f"\n[smoke] turn {turn}: ctx={prompt_tokens}tok in={result.num_input_tokens} "
                  f"out={result.num_output_tokens} verify_rounds={result.verify_count} "
                  f"accept_len={accept_len:.2f} proposal={result.proposal_sum} "
                  f"time={result.gen_time_s:.2f}s tok/s={tok_s:.1f}")
            print(f"[smoke]   text: {result.text[:180]!r}")
            # Any turn that produced no text or no verification round is broken.
            if not result.text.strip() or result.verify_count == 0:
                print(f"[smoke]   FAIL: empty output or no verification round")
                failures += 1
            if result.num_input_tokens != prompt_tokens:
                print(f"[smoke]   FAIL: count_tokens ({prompt_tokens}) != num_input_tokens "
                      f"({result.num_input_tokens})")
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
