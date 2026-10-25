#!/usr/bin/env bash
# Evaluate all supported methods on the fixed TraceSpec-Traj prefixes.
# Native methods use one GPU each. LongSpark uses a separate target/draft pair.
# Model roots, Python environments, GPU pools, and decoding settings are
# configurable through environment variables.

set -euo pipefail
cd "$(dirname "$0")"

read -r -a GPUS_SPEC <<< "${GPUS_SPEC:-0 1 2 3}"
read -r -a GPUS_LONGSPARK <<< "${GPUS_LONGSPARK:-4 5}"
read -r -a METHOD_LIST <<< "${METHODS:-dspark dflash eagle3 dlite longspark}"
read -r -a TARGET_LIST <<< "${TARGETS:-Qwen3-4B Qwen3-8B Qwen3-14B}"

MODEL_ROOT="${MODEL_ROOT:-models}"
DRAFT_ROOT="${DRAFT_ROOT:-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPEC_PYTHON_BIN="${SPEC_PYTHON_BIN:-$PYTHON_BIN}"
LONGSPARK_PYTHON_BIN="${LONGSPARK_PYTHON_BIN:-$PYTHON_BIN}"
MAX_SAMPLES="${MAX_SAMPLES:-2500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-36864}"
TEMPERATURE="${TEMPERATURE:-0.0}"
OUT_ROOT="${OUT_ROOT:-outputs/t${TEMPERATURE}}"

command -v "$SPEC_PYTHON_BIN" >/dev/null 2>&1 || {
  echo "[replay] Native Python executable not found: $SPEC_PYTHON_BIN" >&2
  exit 1
}

native_count=0
for method in "${METHOD_LIST[@]}"; do
  case "$method" in
    dspark|dflash|eagle3|dlite) native_count=$((native_count + 1)) ;;
    longspark) ;;
    *) echo "[replay] Unsupported method: $method" >&2; exit 1 ;;
  esac
done
(( ${#GPUS_SPEC[@]} >= native_count )) || {
  echo "[replay] $native_count native methods require at least $native_count GPUs" >&2
  exit 1
}
if [[ " ${METHOD_LIST[*]} " == *" longspark "* ]]; then
  command -v "$LONGSPARK_PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[replay] LongSpark Python executable not found: $LONGSPARK_PYTHON_BIN" >&2
    exit 1
  }
  (( ${#GPUS_LONGSPARK[@]} == 2 )) || {
    echo "[replay] GPUS_LONGSPARK must contain exactly two GPU indices" >&2
    exit 1
  }
fi

draft_for() {
  local size
  size="$(echo "$2" | tr 'A-Z' 'a-z' | tr '-' '_')"
  case "$1" in
    dspark) echo "dspark_${size}_block7" ;;
    dflash) echo "dflash_${size}_block7" ;;
    eagle3) echo "eagle3_${size}_ttt7" ;;
    dlite) echo "dlite_${size}" ;;
    longspark) echo "LongSpark-$2" ;;
  esac
}

kv_for() {
  case "$1" in
    Qwen3-14B) echo 16 ;;
    Qwen3-8B) echo 24 ;;
    *) echo 32 ;;
  esac
}

mkdir -p "$OUT_ROOT"
for target in "${TARGET_LIST[@]}"; do
  [[ -d "$MODEL_ROOT/$target" ]] || {
    echo "[replay] Target not found: $MODEL_ROOT/$target" >&2
    exit 1
  }
  echo "[replay] Starting $target"
  pids=()
  spec_index=0
  for method in "${METHOD_LIST[@]}"; do
    draft="$(draft_for "$method" "$target")"
    if [[ ! -d "$DRAFT_ROOT/$draft" ]]; then
      echo "[replay] Skipping missing draft: $DRAFT_ROOT/$draft" >&2
      continue
    fi
    out="$OUT_ROOT/$draft"
    mkdir -p "$out"
    log="$out/tracespec-traj-test.log"
    if [[ "$method" == "longspark" ]]; then
      cuda_devices="$(IFS=,; echo "${GPUS_LONGSPARK[*]}")"
      echo "[replay] gpus=$cuda_devices method=$method target=$target"
      CUDA_VISIBLE_DEVICES="$cuda_devices" "$LONGSPARK_PYTHON_BIN" run_agent_eval.py \
        --dataset tracespec-traj --backend longspark \
        --target "$MODEL_ROOT/$target" --draft "$DRAFT_ROOT/$draft" \
        --output-dir "$out" --num-gpus 1 \
        --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-context-tokens "$MAX_CONTEXT_TOKENS" --temperature "$TEMPERATURE" \
        --longspark-target-kv-gib "$(kv_for "$target")" \
        > "$log" 2>&1 &
    else
      gpu="${GPUS_SPEC[$spec_index]}"
      spec_index=$((spec_index + 1))
      echo "[replay] gpu=$gpu method=$method target=$target"
      CUDA_VISIBLE_DEVICES="$gpu" "$SPEC_PYTHON_BIN" run_agent_eval.py \
        --dataset tracespec-traj --backend spec \
        --target "$MODEL_ROOT/$target" --draft "$DRAFT_ROOT/$draft" \
        --output-dir "$out" --num-gpus 1 \
        --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-context-tokens "$MAX_CONTEXT_TOKENS" --temperature "$TEMPERATURE" \
        > "$log" 2>&1 &
    fi
    pids+=("$!")
  done
  (( ${#pids[@]} > 0 )) || {
    echo "[replay] No tasks were launched for $target" >&2
    exit 1
  }
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || {
    echo "[replay] A task failed for $target; inspect $OUT_ROOT" >&2
    exit 1
  }
done

echo "[replay] All targets completed. Results are under $OUT_ROOT"
