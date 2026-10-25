"""DLite model definitions and its TraceSpec evaluator adapter.

The model files come from the DLite inference release under the license in
this directory. The evaluator shares verification and metric accounting with
the other native PyTorch backends.
"""
from .modeling import DLiteDraftModel, DLiteSequentialHead

__all__ = ["DLiteDraftModel", "DLiteSequentialHead"]
