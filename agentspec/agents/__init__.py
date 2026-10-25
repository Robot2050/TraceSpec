from ..config import AgentConfig
from ..datasets import DATASET_BROWSECOMP, DATASET_SWE, DATASET_TRAJ
from .base import Action, LoopAgent


def build_agent(
    dataset: str,
    backend,
    cfg: AgentConfig,
    *,
    search_db_path: str | None = None,
) -> LoopAgent:
    if dataset == DATASET_SWE:
        from .swe_agent import SWEAgent

        return SWEAgent(backend, cfg)
    if dataset == DATASET_BROWSECOMP:
        from .search_agent import SearchAgent

        if search_db_path is None:
            raise ValueError("browsecomp-plus requires search_db_path")
        return SearchAgent(backend, cfg, db_path=search_db_path)
    if dataset == DATASET_TRAJ:
        from .traj_replay_agent import TrajReplayAgent

        return TrajReplayAgent(backend, cfg)
    raise ValueError(f"unknown dataset {dataset!r}")


__all__ = ["Action", "LoopAgent", "build_agent"]
