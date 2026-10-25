"""Run TraceSpec dynamic evaluation or fixed-prefix replay."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from agentspec.agents import build_agent
from agentspec.backends import build_backend
from agentspec.backends.deepspec_backend import pick_free_port
from agentspec.config import AgentConfig, BackendConfig, EvalConfig, PROJECT_ROOT
from agentspec.datasets import (
    DATASET_BROWSECOMP,
    DATASET_SWE,
    DATASET_TRAJ,
    DATASETS,
    load_tasks,
    validate_swe_repos,
)
from agentspec.metrics import aggregate, load_records, write_records
from agentspec.report import build_report


_EXAMPLES = r"""
Dynamic evaluation on SWE-bench Verified:
    CUDA_VISIBLE_DEVICES=0 python run_agent_eval.py \
        --dataset swe-bench-verified --backend spec \
        --target /models/Qwen3-8B \
        --draft /models/dspark_qwen3_8b_block7 \
        --max-samples 100

Fixed-prefix replay on TraceSpec-Traj:
    CUDA_VISIBLE_DEVICES=0 python run_agent_eval.py \
        --dataset tracespec-traj --backend spec \
        --target /models/Qwen3-8B \
        --draft /models/dspark_qwen3_8b_block7 \
        --max-context-tokens 36864

LongSpark replay using one target GPU and one draft GPU:
    CUDA_VISIBLE_DEVICES=0,1 python run_agent_eval.py \
        --dataset tracespec-traj --backend longspark \
        --target /models/Qwen3-8B \
        --draft /models/LongSpark-Qwen3-8B \
        --max-context-tokens 36864

No-GPU pipeline check:
    python run_agent_eval.py --dataset tracespec-traj --backend mock \
        --max-samples 2 --max-new-tokens 32
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TraceSpec measures speculative decoding across agent trajectories.\n"
            "It supports interactive episodes on SWE-bench Verified and "
            "BrowseComp-Plus, plus controlled replay on TraceSpec-Traj. Each run "
            "writes generation records, aggregate metrics, and a formatted report."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_data = parser.add_argument_group("Datasets")
    g_data.add_argument(
        "--dataset", required=True, choices=DATASETS,
        help="Evaluation workload: interactive software engineering, interactive "
             "web research, or fixed-prefix trajectory replay.",
    )
    g_data.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "datasets"),
        help="Dataset root containing swe-bench-verified, browsecomp-plus, and "
             "tracespec-traj (default: %(default)s).",
    )
    g_data.add_argument(
        "--skip-repo-check", action="store_true",
        help="Skip the preflight check for SWE repositories and base commits. "
             "Invalid instances then finish with an error during execution.",
    )

    g_traj = parser.add_argument_group("TraceSpec-Traj filters")
    g_traj.add_argument(
        "--traj-phase", default=None,
        help="Comma-separated phases: plan, act, reflect, answer.",
    )
    g_traj.add_argument(
        "--traj-bucket", default=None,
        help="Comma-separated nominal context buckets: 2k, 4k, 8k, 16k, 32k.",
    )
    g_traj.add_argument(
        "--traj-mode", default=None,
        help="Comma-separated workload filters: browse, swe.",
    )

    g_model = parser.add_argument_group("Models and backends")
    g_model.add_argument(
        "--backend", default="spec", choices=["spec", "longspark", "ar", "mock"],
        help="spec runs DFlash, DSpark, EAGLE-3, or DLite in the native PyTorch "
             "backend; longspark uses the split SGLang runtime; ar is the "
             "autoregressive target baseline; mock checks the pipeline without "
             "a GPU (default: %(default)s).",
    )
    g_model.add_argument(
        "--target", default="",
        help="Target model path or Hugging Face identifier. Required by spec, "
             "longspark, and ar.",
    )
    g_model.add_argument(
        "--draft", default=None,
        help="Draft checkpoint path or Hugging Face identifier. Required by "
             "spec and longspark. The spec backend detects the architecture "
             "from config.json.",
    )
    g_model.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of sharded workers. spec and ar use one GPU per worker; "
             "longspark uses target_tp + 1 GPUs per worker (default: %(default)s).",
    )

    g_agent = parser.add_argument_group("Agent loop")
    g_agent.add_argument(
        "--max-samples", type=int, default=None,
        help="Number of tasks in a seeded subset. By default, use every task "
             "after applying replay filters.",
    )
    g_agent.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum tool-action steps per dynamic episode (default: %(default)s).",
    )
    g_agent.add_argument(
        "--max-new-tokens", type=int, default=4096,
        help="Maximum generated tokens per model call (default: %(default)s).",
    )
    g_agent.add_argument(
        "--max-context-tokens", type=int, default=32768,
        help="Prompt token limit. Dynamic episodes trim old intermediate turns; "
             "replay skips states above the limit (default: %(default)s).",
    )
    g_agent.add_argument(
        "--reflect-every", type=int, default=5,
        help="Insert a reflection call every N action steps; 0 disables periodic "
             "reflection (default: %(default)s).",
    )

    g_dec = parser.add_argument_group("Decoding")
    g_dec.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature shared by target and draft (default: %(default)s).",
    )
    g_dec.add_argument(
        "--confidence-threshold", type=float, default=0.0,
        help="DSpark proposal truncation threshold in [0, 1]; 0 keeps complete "
             "blocks (default: %(default)s).",
    )
    g_dec.add_argument(
        "--seed", type=int, default=980406,
        help="Seed for task sampling and generation (default: %(default)s).",
    )

    g_ls = parser.add_argument_group("LongSpark backend")
    g_ls.add_argument(
        "--longspark-target-devices", default="",
        help="Comma-separated logical GPU indices for the target. By default, "
             "devices are allocated consecutively for each worker.",
    )
    g_ls.add_argument(
        "--longspark-draft-device", type=int, default=-1,
        help="Logical GPU index for the drafter; -1 selects the device after "
             "the target ranks (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-target-tp", type=int, default=1, choices=[1, 2, 4],
        help="Target tensor-parallel degree (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-target-kv-gib", type=float, default=32.0,
        help="KV-cache budget per target rank in GiB (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-draft-state-gib", type=float, default=8.0,
        help="Draft state-pool budget in GiB (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-context-length", type=int, default=40968,
        help="Serving context length; use 40968 with native or 140288 with "
             "fixed_ntk (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-position-variant", default="native", choices=["native", "fixed_ntk"],
        help="Position-encoding variant for the selected context length "
             "(default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-prefill-chunk-size", type=int, default=8192,
        help="Prefill chunk size (default: %(default)s).",
    )
    g_ls.add_argument(
        "--longspark-eos-token-ids", default="",
        help="Optional comma-separated stop token IDs. By default, use the "
             "target tokenizer EOS and standard Qwen end tokens.",
    )

    g_tool = parser.add_argument_group("Tools")
    g_tool.add_argument(
        "--observation-max-chars", type=int, default=6000,
        help="Maximum characters returned by one tool observation "
             "(default: %(default)s).",
    )
    g_tool.add_argument(
        "--search-top-k", type=int, default=5,
        help="BM25 snippets returned per BrowseComp-Plus search "
             "(default: %(default)s).",
    )
    g_tool.add_argument(
        "--search-index-dir", default="",
        help="FTS5 index directory. By default, use .tracespec_cache under "
             "the data root and build the index on first use.",
    )

    g_out = parser.add_argument_group("Output")
    g_out.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs"),
        help="Root directory for evaluation runs (default: %(default)s).",
    )
    g_out.add_argument(
        "--run-name", default=None,
        help="Run directory name. Each run contains records, summary.json, "
             "report.txt, and config.json.",
    )
    return parser.parse_args()


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        backend=BackendConfig(
            backend=args.backend,
            target_name_or_path=args.target,
            draft_name_or_path=args.draft,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            confidence_threshold=args.confidence_threshold,
            seed=args.seed,
            longspark_target_devices=args.longspark_target_devices,
            longspark_draft_device=args.longspark_draft_device,
            longspark_target_tp=args.longspark_target_tp,
            longspark_target_kv_gib=args.longspark_target_kv_gib,
            longspark_draft_state_gib=args.longspark_draft_state_gib,
            longspark_context_length=args.longspark_context_length,
            longspark_position_variant=args.longspark_position_variant,
            longspark_prefill_chunk_size=args.longspark_prefill_chunk_size,
            longspark_eos_token_ids=args.longspark_eos_token_ids,
        ),
        agent=AgentConfig(
            max_steps=args.max_steps,
            reflect_every=args.reflect_every,
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            observation_max_chars=args.observation_max_chars,
            search_top_k=args.search_top_k,
            search_index_dir=args.search_index_dir,
        ),
        traj_split="test",
        traj_phase_filter=_split_csv(args.traj_phase) or [],
        traj_bucket_filter=_split_csv(args.traj_bucket) or [],
        traj_mode_filter=_split_csv(args.traj_mode) or [],
    )


def run_shard(local_rank: int, world_size: int, cfg: EvalConfig, run_dir: Path) -> None:
    """Worker: run this rank's share of episodes and write one record shard."""
    import torch

    # Spawned workers are fresh interpreters: re-configure logging here so the
    # agent-loop instrumentation is visible in multi-GPU runs too.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s][rank{local_rank}] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    tasks = load_tasks(
        cfg.dataset, cfg.data_root, cfg.max_samples, cfg.backend.seed,
        traj_split=cfg.traj_split,
        traj_phase_filter=cfg.traj_phase_filter,
        traj_bucket_filter=cfg.traj_bucket_filter,
        traj_mode_filter=cfg.traj_mode_filter,
    )
    my_tasks = tasks[local_rank::world_size]

    search_db_path = None
    if cfg.dataset == DATASET_BROWSECOMP:
        index_dir = cfg.agent.search_index_dir or str(
            Path(cfg.data_root) / ".tracespec_cache"
        )
        from agentspec.tools.search_tools import index_path

        search_db_path = str(index_path(index_dir))

    script = (
        "search" if cfg.dataset == DATASET_BROWSECOMP
        else "traj" if cfg.dataset == DATASET_TRAJ
        else "swe"
    )
    backend = build_backend(
        cfg.backend,
        local_rank=local_rank,
        script=script,
    )
    agent = build_agent(cfg.dataset, backend, cfg.agent, search_db_path=search_db_path)

    records = []
    # Give each worker a separate progress line and throttle log updates.
    progress = tqdm(
        my_tasks,
        desc=f"[tracespec] {cfg.dataset} rank{local_rank}",
        unit="ep",
        position=local_rank,
        mininterval=10.0,
        disable=False,
    )
    for i, task in enumerate(progress):
        episode_id = f"{cfg.dataset}:{task.task_id}:r{local_rank}"
        if local_rank == 0:
            print(f"[tracespec] episode {i + 1}/{len(my_tasks)}: {task.task_id}", flush=True)
        t0 = time.perf_counter()
        record = agent.run(task, episode_id)
        records.append(record)
        gens = record.generations
        rounds = sum(g.verify_count for g in gens)
        acc = sum(g.accepted_sum for g in gens) / rounds if rounds else 0.0
        progress.set_postfix_str(
            f"{record.finish_reason}, steps={record.num_steps}, acc={acc:.2f}",
            refresh=False,
        )
        if local_rank == 0:
            print(
                f"[tracespec]   -> {record.finish_reason}, steps={record.num_steps}, "
                f"accept_len={acc:.3f}, time={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    progress.close()

    write_records(records, run_dir / "records" / f"{cfg.dataset}.rank{local_rank}.jsonl")
    backend.clean_up()
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    cfg = build_eval_config(args)

    run_name = args.run_name or (
        f"{cfg.dataset}-{cfg.traj_split}-{cfg.backend.backend}-{time.strftime('%Y%m%d-%H%M%S')}"
        if cfg.dataset == DATASET_TRAJ
        else f"{cfg.dataset}-{cfg.backend.backend}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Fail fast on broken SWE checkouts, before any model is loaded: every
    # sampled instance must have its repo present with its base_commit
    # reachable, otherwise its episode would run against the wrong code.
    if cfg.dataset == DATASET_SWE and not args.skip_repo_check:
        tasks = load_tasks(cfg.dataset, cfg.data_root, cfg.max_samples, cfg.backend.seed)
        problems = validate_swe_repos(tasks)
        if problems:
            for problem in problems:
                logging.error(f"[repo-check] {problem}")
            sys.exit(
                f"[repo-check] {len(problems)} problem(s) found — fix the checkouts under "
                f"{Path(cfg.data_root) / DATASET_SWE / 'repos'} (full-history clones), "
                f"or rerun with --skip-repo-check"
            )
        repos = {task.meta["repo_dir"] for task in tasks}
        logging.info(
            f"[repo-check] ok: {len(tasks)} sampled instances across {len(repos)} repos"
        )

    # Build the search index once, before workers fork.
    if cfg.dataset == DATASET_BROWSECOMP:
        index_dir = cfg.agent.search_index_dir or str(
            Path(cfg.data_root) / ".tracespec_cache"
        )
        from agentspec.tools.search_tools import ensure_index

        ensure_index(Path(cfg.data_root) / DATASET_BROWSECOMP / "corpus" / "documents.jsonl",
                     index_dir)

    # The spec backend's NCCL rendezvous needs one fixed port per run. Pin a
    # free one here, in the parent, before workers spawn (they inherit the
    # environment), so concurrent runs on this machine never collide.
    # MASTER_ADDR/MASTER_PORT already set in the environment are honored.
    if cfg.backend.backend == "spec":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(pick_free_port())
        logging.info(
            f"[dist] rendezvous: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )

    world_size = max(1, cfg.num_gpus)
    if world_size == 1:
        run_shard(0, 1, cfg, run_dir)
    else:
        import torch

        torch.multiprocessing.spawn(
            run_shard, args=(world_size, cfg, run_dir), nprocs=world_size
        )

    records = load_records(run_dir / "records")
    summary = aggregate(records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = build_report(
        summary,
        title=f"TraceSpec: {cfg.dataset} | backend={cfg.backend.backend} | "
              f"target={cfg.backend.target_name_or_path} | draft={cfg.backend.draft_name_or_path}",
    )
    print("\n" + report, flush=True)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(f"\n[tracespec] records + summary written to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
