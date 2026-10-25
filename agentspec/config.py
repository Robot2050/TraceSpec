"""Configuration for the TraceSpec evaluation harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTEXT_BUCKETS: list[tuple[int, int | None, str]] = [
    (0, 2048, "0-2k"),
    (2048, 4096, "2-4k"),
    (4096, 8192, "4-8k"),
    (8192, 16384, "8-16k"),
    (16384, 32768, "16-32k"),
    (32768, None, "32k+"),
]

MODE_PLAN = "plan"
MODE_ACT = "act"
MODE_REFLECT = "reflect"
MODE_ANSWER = "answer"
AGENT_MODES = (MODE_PLAN, MODE_ACT, MODE_REFLECT, MODE_ANSWER)


def context_bucket(num_tokens: int) -> str:
    for lo, hi, label in CONTEXT_BUCKETS:
        if num_tokens >= lo and (hi is None or num_tokens < hi):
            return label
    return CONTEXT_BUCKETS[-1][2]


STEP_BUCKETS: list[tuple[str, int, int | None]] = [
    ("plan", -1, 0),
    ("0-4", 0, 5),
    ("5-9", 5, 10),
    ("10-19", 10, 20),
    ("20+", 20, None),
]


def step_bucket(step: int) -> str:
    for label, lo, hi in STEP_BUCKETS:
        if step >= lo and (hi is None or step < hi):
            return label
    return STEP_BUCKETS[-1][0]


@dataclass
class BackendConfig:
    """Model and decoding settings shared by all backends."""

    backend: str = "spec"
    target_name_or_path: str = ""
    draft_name_or_path: str | None = None
    max_new_tokens: int = 4096
    temperature: float = 0.0
    tree_draft_temperature: float = 0.7
    confidence_threshold: float = 0.0
    seed: int = 980406

    # Each LongSpark worker uses target_tp target devices and one draft device.
    longspark_target_devices: str = ""
    longspark_draft_device: int = -1
    longspark_target_tp: int = 1
    longspark_target_kv_gib: float = 32.0
    longspark_draft_state_gib: float = 8.0
    longspark_context_length: int = 40968
    longspark_position_variant: str = "native"
    longspark_prefill_chunk_size: int = 8192
    longspark_eos_token_ids: str = ""


@dataclass
class AgentConfig:
    """Controls interaction, context handling, and tool observations."""

    max_steps: int = 50
    reflect_every: int = 5
    max_context_tokens: int = 32768
    max_new_tokens: int = 4096
    observation_max_chars: int = 6000
    command_timeout_s: int = 60
    search_top_k: int = 5
    search_index_dir: str = ""


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    dataset: str = "swe-bench-verified"
    data_root: str = str(PROJECT_ROOT / "datasets")
    max_samples: int | None = None
    output_dir: str = str(PROJECT_ROOT / "outputs")
    num_gpus: int = 1
    backend: BackendConfig = field(default_factory=BackendConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    traj_split: str = "test"
    traj_phase_filter: list[str] = field(default_factory=list)
    traj_bucket_filter: list[str] = field(default_factory=list)
    traj_mode_filter: list[str] = field(default_factory=list)
