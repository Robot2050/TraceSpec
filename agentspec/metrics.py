"""Per-generation and per-episode metric records, plus aggregation.

Accepted length: tokens committed per
verification round (accepted draft tokens + 1 bonus token), averaged over
*all proposal rounds* (not per request) — i.e. sum(accepted) / sum(rounds).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CONTEXT_BUCKETS, AGENT_MODES, STEP_BUCKETS, context_bucket, step_bucket


@dataclass
class GenerationRecord:
    """One model call (one agent turn)."""

    episode_id: str
    step: int
    mode: str                     # plan | act | reflect | answer
    num_input_tokens: int         # context length at this call
    context_bucket: str
    num_output_tokens: int
    verify_count: int             # number of target verification rounds
    accepted_sum: int             # sum of per-round acceptance lengths
    proposal_sum: int             # sum of per-round effective proposal lengths
    gen_time_s: float             # wall time of the whole call (prefill + decode)
    backend: str                  # spec | longspark | ar | mock

    @property
    def acceptance_length(self) -> float:
        return self.accepted_sum / self.verify_count if self.verify_count else 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.num_output_tokens / self.gen_time_s if self.gen_time_s > 0 else 0.0


@dataclass
class EpisodeRecord:
    """One agent episode (one dataset instance)."""

    episode_id: str
    dataset: str
    task_id: str
    finish_reason: str            # submitted | stuck_format | stuck_loop | context_limit | forced_answer | error
    num_steps: int                # number of tool-call steps executed
    total_time_s: float           # end-to-end wall time of the episode
    model_time_s: float           # time inside model calls
    tool_time_s: float            # time inside tool execution
    truncations: int              # how many times context had to be trimmed
    replans: int = 0              # reflect + re-plan interventions after stalls
    output: str = ""              # final answer / submitted summary (truncated)
    generations: list[GenerationRecord] = field(default_factory=list)


class EpisodeRecorder:
    """Collects records for one episode; the agent calls `add_generation`
    after every backend call."""

    def __init__(self, episode_id: str, dataset: str, task_id: str):
        self.episode = EpisodeRecord(
            episode_id=episode_id,
            dataset=dataset,
            task_id=task_id,
            finish_reason="error",
            num_steps=0,
            total_time_s=0.0,
            model_time_s=0.0,
            tool_time_s=0.0,
            truncations=0,
            output="",
        )
        self._t0 = time.perf_counter()

    def add_generation(
        self,
        *,
        step: int,
        mode: str,
        num_input_tokens: int,
        num_output_tokens: int,
        verify_count: int,
        accepted_sum: int,
        proposal_sum: int,
        gen_time_s: float,
        backend: str,
    ) -> None:
        self.episode.generations.append(
            GenerationRecord(
                episode_id=self.episode.episode_id,
                step=step,
                mode=mode,
                num_input_tokens=num_input_tokens,
                context_bucket=context_bucket(num_input_tokens),
                num_output_tokens=num_output_tokens,
                verify_count=verify_count,
                accepted_sum=accepted_sum,
                proposal_sum=proposal_sum,
                gen_time_s=gen_time_s,
                backend=backend,
            )
        )
        self.episode.model_time_s += gen_time_s

    def finish(self, *, finish_reason: str, output: str = "") -> EpisodeRecord:
        self.episode.finish_reason = finish_reason
        self.episode.output = output[:2000]
        self.episode.total_time_s = time.perf_counter() - self._t0
        self.episode.tool_time_s = max(
            0.0, self.episode.total_time_s - self.episode.model_time_s
        )
        return self.episode


def write_records(records: list[EpisodeRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_records(pattern_dir: str | Path) -> list[EpisodeRecord]:
    """Load every `*.jsonl` record shard under a directory."""
    records: list[EpisodeRecord] = []
    for path in sorted(Path(pattern_dir).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                generations = [GenerationRecord(**g) for g in payload.pop("generations")]
                records.append(EpisodeRecord(generations=generations, **payload))
    return records


def _mean_acceptance(generations: list[GenerationRecord]) -> float:
    rounds = sum(g.verify_count for g in generations)
    if rounds == 0:
        return 0.0
    return sum(g.accepted_sum for g in generations) / rounds


def _mean_verify_rate(generations: list[GenerationRecord]) -> float:
    rounds = sum(g.verify_count for g in generations)
    proposed = sum(g.proposal_sum for g in generations)
    if rounds == 0 or proposed + rounds == 0:
        return 0.0
    return sum(g.accepted_sum for g in generations) / (proposed + rounds)


def aggregate(records: list[EpisodeRecord]) -> dict:
    """Build the report payload: overall, per-context-bucket, per-mode and
    latency statistics."""
    all_gens = [g for ep in records for g in ep.generations]

    by_bucket: dict[str, list[GenerationRecord]] = {}
    for _, _, label in CONTEXT_BUCKETS:
        by_bucket[label] = []
    for gen in all_gens:
        by_bucket.setdefault(gen.context_bucket, []).append(gen)

    by_mode: dict[str, list[GenerationRecord]] = {m: [] for m in AGENT_MODES}
    for gen in all_gens:
        by_mode.setdefault(gen.mode, []).append(gen)

    by_step: dict[str, list[GenerationRecord]] = {label: [] for label, _, _ in STEP_BUCKETS}
    for gen in all_gens:
        by_step[step_bucket(gen.step)].append(gen)

    # Cross-tabulate context and progress so comparisons can hold the episode
    # stage fixed. Planning calls are excluded because they have step -1.
    progress_bands = [label for label, lo, _ in STEP_BUCKETS if lo >= 0]
    by_context_step: dict[str, dict[str, list[GenerationRecord]]] = {
        label: {band: [] for band in progress_bands}
        for _, _, label in CONTEXT_BUCKETS
    }
    for gen in all_gens:
        if gen.step < 0:
            continue
        by_context_step[gen.context_bucket][step_bucket(gen.step)].append(gen)

    total_times = sorted(ep.total_time_s for ep in records)
    n = len(total_times)
    median_total = total_times[n // 2] if n else 0.0

    return {
        "num_episodes": len(records),
        "finish_reasons": _count_by(records, lambda ep: ep.finish_reason),
        "overall": {
            "num_generations": len(all_gens),
            "acceptance_length": _mean_acceptance(all_gens),
            "verify_rate": _mean_verify_rate(all_gens),
            "mean_output_tokens": _safe_mean([g.num_output_tokens for g in all_gens]),
            "mean_input_tokens": _safe_mean([g.num_input_tokens for g in all_gens]),
            "max_input_tokens": max((g.num_input_tokens for g in all_gens), default=0),
        },
        "by_context_bucket": {
            label: {
                "num_generations": gens and len(gens) or 0,
                "acceptance_length": _mean_acceptance(gens),
                "mean_output_tokens": _safe_mean([g.num_output_tokens for g in gens]),
                "mean_gen_time_s": _safe_mean([g.gen_time_s for g in gens]),
                "mean_tokens_per_s": _safe_mean([g.tokens_per_s for g in gens]),
            }
            for label, gens in by_bucket.items()
        },
        "by_mode": {
            mode: {
                "num_generations": len(gens),
                "acceptance_length": _mean_acceptance(gens),
                "mean_input_tokens": _safe_mean([g.num_input_tokens for g in gens]),
                "mean_output_tokens": _safe_mean([g.num_output_tokens for g in gens]),
            }
            for mode, gens in by_mode.items()
        },
        "by_step": {
            label: {
                "num_generations": len(gens),
                "acceptance_length": _mean_acceptance(gens),
                "mean_input_tokens": _safe_mean([g.num_input_tokens for g in gens]),
                "mean_output_tokens": _safe_mean([g.num_output_tokens for g in gens]),
            }
            for label, gens in by_step.items()
        },
        "by_context_step": {
            label: {
                band: {
                    "num_generations": len(gens),
                    "acceptance_length": _mean_acceptance(gens),
                    "verify_rounds": sum(g.verify_count for g in gens),
                }
                for band, gens in bands.items()
            }
            for label, bands in by_context_step.items()
        },
        "latency": {
            "mean_episode_time_s": _safe_mean([ep.total_time_s for ep in records]),
            "median_episode_time_s": median_total,
            "total_time_s": sum(total_times),
            "mean_model_time_s": _safe_mean([ep.model_time_s for ep in records]),
            "mean_tool_time_s": _safe_mean([ep.tool_time_s for ep in records]),
            "mean_steps_per_episode": _safe_mean([ep.num_steps for ep in records]),
            "mean_replans_per_episode": _safe_mean([ep.replans for ep in records]),
            "overall_tokens_per_s": (
                sum(g.num_output_tokens for g in all_gens)
                / sum(g.gen_time_s for g in all_gens)
                if sum(g.gen_time_s for g in all_gens) > 0
                else 0.0
            ),
        },
    }


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_by(records, key) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        k = key(record)
        counts[k] = counts.get(k, 0) + 1
    return counts
