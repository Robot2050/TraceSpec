#!/usr/bin/env bash
# Install the separate LongSpark runtime environment.
# Use PYTHON_BIN to modify an existing environment, or ENV_NAME to create a
# Conda environment. The native evaluators and LongSpark must not share an
# environment because their PyTorch ABI requirements differ.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d agentspec/vendor/sglang/python ]]; then
  echo "[install] Extract agentspec/vendor/sglang.tar.gz into agentspec/vendor/ before installation" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
ENV_NAME="${ENV_NAME:-longspark}"

if [[ -z "$PYTHON_BIN" ]]; then
  command -v conda >/dev/null 2>&1 || {
    echo "[install] Conda is required when PYTHON_BIN is not set" >&2
    exit 1
  }
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[install] Creating Conda environment $ENV_NAME with Python 3.11"
    conda create -n "$ENV_NAME" python=3.11 -y
  else
    echo "[install] Reusing Conda environment $ENV_NAME"
  fi
fi

[[ -x "$PYTHON_BIN" ]] || {
  echo "[install] Python executable not found: $PYTHON_BIN" >&2
  exit 1
}

echo "[install] Target interpreter: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install -q --upgrade pip setuptools wheel

if ! "$PYTHON_BIN" -c 'import torch; assert torch.__version__.startswith("2.11")' >/dev/null 2>&1; then
  echo "[install] Installing torch 2.11.0+cu128"
  "$PYTHON_BIN" -m pip install torch==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
fi

echo "[install] Installing SGLang runtime prerequisites"
"$PYTHON_BIN" -m pip install \
  orjson aiohttp fastapi uvicorn requests pydantic psutil uvloop zmq pybase64 \
  sentencepiece tiktoken einops ninja accelerate \
  IPython partial_json_parser msgspec compressed-tensors gguf numba \
  diskcache outlines xgrammar setproctitle distro easydict

# Avoid dependency resolution replacing the required PyTorch build.
"$PYTHON_BIN" -m pip install --no-deps torchvision
"$PYTHON_BIN" -m pip install --no-deps sglang-kernel==0.4.4
"$PYTHON_BIN" -m pip install --no-deps flashinfer-python==0.6.3 flashinfer-cubin==0.6.3
"$PYTHON_BIN" -m pip install --no-deps apache-tvm-ffi==0.1.12 torch-c-dlpack-ext==0.1.5
"$PYTHON_BIN" -m pip install transformers==5.12.1
"$PYTHON_BIN" -m pip install tqdm prettytable

echo "[install] Verifying the runtime"
PYTHONPATH=agentspec/vendor/sglang/python:agentspec/vendor \
  "$PYTHON_BIN" -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
