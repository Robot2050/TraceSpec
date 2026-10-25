from ..config import BackendConfig
from .base import BaseBackend, GenerationResult


def build_backend(cfg: BackendConfig, local_rank: int = 0, script: str = "swe") -> BaseBackend:
    if cfg.backend == "spec":
        from .deepspec_backend import DeepSpecBackend

        return DeepSpecBackend(cfg, local_rank=local_rank)
    if cfg.backend == "longspark":
        from .longspark_backend import LongSparkBackend

        return LongSparkBackend(cfg, local_rank=local_rank)
    if cfg.backend == "ar":
        from .autoregressive_backend import AutoRegressiveBackend

        return AutoRegressiveBackend(cfg, local_rank=local_rank)
    if cfg.backend == "mock":
        from .mock_backend import MockBackend

        return MockBackend(cfg, local_rank=local_rank, script=script)
    raise ValueError(f"unknown backend {cfg.backend!r} (expected spec|longspark|ar|mock)")


__all__ = ["BaseBackend", "GenerationResult", "build_backend"]
