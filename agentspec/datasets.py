"""Unified dataset access for dynamic execution and fixed-prefix replay."""
from __future__ import annotations

import gzip
import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DATASET_SWE = "swe-bench-verified"
DATASET_BROWSECOMP = "browsecomp-plus"
DATASET_TRAJ = "tracespec-traj"
DATASETS = (DATASET_SWE, DATASET_BROWSECOMP, DATASET_TRAJ)


@dataclass
class AgentTask:
    task_id: str
    dataset: str
    query: str
    reference: str | None
    meta: dict = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL directly or from a gzip-compressed sibling."""
    if not path.is_file() and path.suffix != ".gz":
        compressed_path = path.with_name(path.name + ".gz")
        if compressed_path.is_file():
            path = compressed_path
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    rows = []
    with handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(
    dataset: str,
    data_root: str,
    max_samples: int | None = None,
    seed: int = 980406,
    *,
    traj_split: str = "test",
    traj_phase_filter: list[str] | None = None,
    traj_bucket_filter: list[str] | None = None,
    traj_mode_filter: list[str] | None = None,
) -> list[AgentTask]:
    """Load one dataset, optionally down-sampled with a seeded shuffle."""
    root = Path(data_root)
    if dataset == DATASET_TRAJ:
        return load_traj_tasks(
            data_root,
            split=traj_split,
            max_samples=max_samples,
            seed=seed,
            phase_filter=traj_phase_filter,
            bucket_filter=traj_bucket_filter,
            mode_filter=traj_mode_filter,
        )
    if dataset == DATASET_SWE:
        rows = _read_jsonl(root / DATASET_SWE / "instances.jsonl")
        tasks = []
        for row in rows:
            repo_dir = root / DATASET_SWE / "repos" / row["repo"].replace("/", "_")
            tasks.append(
                AgentTask(
                    task_id=row["instance_id"],
                    dataset=dataset,
                    query=row["problem_statement"],
                    reference=row.get("patch"),
                    meta={
                        "repo": row["repo"],
                        "repo_dir": str(repo_dir),
                        "base_commit": row["base_commit"],
                        "version": row.get("version", ""),
                    },
                )
            )
    elif dataset == DATASET_BROWSECOMP:
        rows = _read_jsonl(root / DATASET_BROWSECOMP / "queries.jsonl")
        corpus = root / DATASET_BROWSECOMP / "corpus" / "documents.jsonl"
        tasks = [
            AgentTask(
                task_id=str(row["query_id"]),
                dataset=dataset,
                query=row["query"],
                reference=row.get("answer"),
                meta={"corpus_path": str(corpus)},
            )
            for row in rows
        ]
    else:
        raise ValueError(f"unknown dataset: {dataset!r} (expected one of {DATASETS})")

    if max_samples is not None and len(tasks) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(tasks)
        tasks = tasks[:max_samples]
    return tasks


def load_traj_tasks(
    data_root: str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 980406,
    phase_filter: list[str] | None = None,
    bucket_filter: list[str] | None = None,
    mode_filter: list[str] | None = None,
) -> list[AgentTask]:
    """Load TraceSpec-Traj states, with optional state-attribute filters.

    Each state contains a complete message prefix and the assistant turn that
    followed it. Replay presents only the prefix to the selected backend.
    """
    path = Path(data_root) / DATASET_TRAJ / f"{split}.jsonl"
    rows = _read_jsonl(path)
    if phase_filter:
        rows = [r for r in rows if r.get("phase") in set(phase_filter)]
    if bucket_filter:
        rows = [r for r in rows if r.get("length_bucket") in set(bucket_filter)]
    if mode_filter:
        rows = [r for r in rows if r.get("mode") in set(mode_filter)]
    tasks = [
        AgentTask(
            task_id=r["sample_id"],
            dataset=DATASET_TRAJ,
            query="",                      # the full input lives in meta
            reference=json.dumps(r.get("output", {}), ensure_ascii=False),
            meta={
                "split": r.get("split", split),
                "mode": r.get("mode", ""),
                "phase": r.get("phase", ""),
                "length_bucket": r.get("length_bucket", ""),
                "prefix_tokens_est": r.get("prefix_tokens_est", 0),
                "num_tool_rounds": r.get("num_tool_rounds", 0),
                "source": r.get("source", {}),
                "input": r["input"],
                "output": r.get("output", {}),
            },
        )
        for r in rows
    ]
    if max_samples is not None and len(tasks) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(tasks)
        tasks = tasks[:max_samples]
    return tasks


def validate_swe_repos(tasks: list[AgentTask]) -> list[str]:
    """Pre-flight check for SWE-bench tasks: every sampled instance's repo
    directory must exist, be a git repository, and contain its base_commit
    (i.e. the clone must carry full history, not be shallow).

    Returns a list of human-readable problems; an empty list means every
    sampled instance can set up its worktree. Checking before model loading
    avoids discovering a broken checkout one expensive episode at a time.
    """
    by_repo: dict[str, list[AgentTask]] = {}
    for task in tasks:
        by_repo.setdefault(task.meta["repo_dir"], []).append(task)

    problems: list[str] = []
    for repo_dir, repo_tasks in by_repo.items():
        if not Path(repo_dir).is_dir():
            problems.append(
                f"repo dir missing: {repo_dir} "
                f"({len(repo_tasks)} instances, e.g. {repo_tasks[0].task_id})"
            )
            continue
        try:
            probe = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "--git-dir"],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            return ["git executable not found on PATH"]
        except subprocess.TimeoutExpired:
            problems.append(f"git probe timed out: {repo_dir}")
            continue
        if probe.returncode != 0:
            problems.append(
                f"not a git repository: {repo_dir} ({len(repo_tasks)} instances)"
            )
            continue
        missing = []
        for task in repo_tasks:
            commit = task.meta["base_commit"]
            try:
                ok = subprocess.run(
                    ["git", "-C", repo_dir, "cat-file", "-e", f"{commit}^{{commit}}"],
                    capture_output=True, timeout=30,
                )
                present = ok.returncode == 0
            except subprocess.TimeoutExpired:
                present = False
            if not present:
                missing.append(f"{task.task_id} ({commit[:12]})")
        if missing:
            shown = ", ".join(missing[:5])
            if len(missing) > 5:
                shown += f" ... and {len(missing) - 5} more"
            problems.append(
                f"{repo_dir}: {len(missing)}/{len(repo_tasks)} base commits absent "
                f"(shallow clone?): {shown}"
            )
    return problems
