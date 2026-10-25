from .common import Eagle3ForwardOutput, extract_eagle3_context_feature

try:
    from .gemma4 import Gemma4Eagle3Model
except ModuleNotFoundError:
    # transformers 版本不含 gemma4 模块时仍可使用 Qwen3 draft；
    # 只有实际加载 Gemma4 架构的 checkpoint 才会需要它。
    Gemma4Eagle3Model = None

from .qwen3 import Qwen3Eagle3Model

__all__ = [
    "Eagle3ForwardOutput",
    "Gemma4Eagle3Model",
    "Qwen3Eagle3Model",
    "extract_eagle3_context_feature",
]
