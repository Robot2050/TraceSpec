from .common import DSparkForwardOutput, extract_context_feature

try:
    from .gemma4 import Gemma4DSparkModel
except ModuleNotFoundError:
    # transformers 版本不含 gemma4 模块时仍可使用 Qwen3 draft；
    # 只有实际加载 Gemma4 架构的 checkpoint 才会需要它。
    Gemma4DSparkModel = None

from .qwen3 import Qwen3DSparkModel

__all__ = [
    "DSparkForwardOutput",
    "extract_context_feature",
    "Gemma4DSparkModel",
    "Qwen3DSparkModel",
]
