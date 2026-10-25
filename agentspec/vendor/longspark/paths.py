"""Locations of the vendored LongSpark runtime inside TraceSpec.

Upstream LongSpark keeps `src/longspark`, `vendor/sglang` and
`vendor/specforge` under one repository root.  Inside TraceSpec they live
under `agentspec/vendor/`, so that directory plays the role of the repository
root: the `longspark`, `sglang` and `specforge` packages all sit directly in it.
"""
from pathlib import Path

# .../agentspec/vendor/longspark/paths.py -> .../agentspec/vendor
VENDOR_ROOT = Path(__file__).resolve().parents[1]

# Kept for compatibility with upstream modules that still import these names.
REPO_ROOT = VENDOR_ROOT
SRC_DIR = VENDOR_ROOT

# The vendored SGLang runtime and the SpecForge draft-model definitions are
# imported from the source tree rather than installed as packages.
VENDOR_PATHS = (VENDOR_ROOT / "sglang" / "python", VENDOR_ROOT)
