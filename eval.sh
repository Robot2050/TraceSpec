#!/usr/bin/env bash
# Run native PyTorch evaluators on dynamic agent workloads.
# Configure GPUS, METHODS, TARGETS, DATASETS, MODEL_ROOT, and DRAFT_ROOT
# through the environment. Each task uses one GPU and writes its own log.

set -euo pipefail
cd "$(dirname "$0")"

read -r -a GPU_LIST <<< "${GPUS:-0 1}"
read -r -a METHOD_LIST <<< "${METHODS:-dspark}"
read -r -a TARGET_LIST <<< "${TARGETS:-Qwen3-8B}"
read -r -a DATASET_LIST <<< "${DATASETS:-browsecomp-plus swe-bench-verified}"

MODEL_ROOT="${MODEL_ROOT:-models}"
DRAFT_ROOT="${DRAFT_ROOT:-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
OUT_ROOT="${OUT_ROOT:-outputs/t${TEMPERATURE}}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "[eval] Python executable not found: $PYTHON_BIN" >&2
  exit 1
}

draft_for() {
  local size
  size="$(echo "$2" | tr 'A-Z' 'a-z' | tr '-' '_')"
  case "$1" in
    dspark) echo "dspark_${size}_block7" ;;
    dflash) echo "dflash_${size}_block7" ;;
    eagle3) echo "eagle3_${size}_ttt7" ;;
    dlite) echo "dlite_${size}" ;;
    *) echo "[eval] Unsupported method: $1" >&2; return 1 ;;
  esac
}

task_count=$(( ${#METHOD_LIST[@]} * ${#TARGET_LIST[@]} * ${#DATASET_LIST[@]} ))
(( ${#GPU_LIST[@]} >= task_count )) || {
  echo "[eval] $task_count tasks require at least $task_count GPUs; received ${#GPU_LIST[@]}" >&2
  exit 1
}

pids=()
i=0
for method in "${METHOD_LIST[@]}"; do
  for target in "${TARGET_LIST[@]}"; do
    draft="$(draft_for "$method" "$target")"
    [[ -d "$MODEL_ROOT/$target" ]] || {
      echo "[eval] Target not found: $MODEL_ROOT/$target" >&2
      exit 1
    }
    if [[ ! -d "$DRAFT_ROOT/$draft" ]]; then
      echo "[eval] Skipping missing draft: $DRAFT_ROOT/$draft" >&2
      continue
    fi
    out="$OUT_ROOT/$draft"
    mkdir -p "$out"
    for dataset in "${DATASET_LIST[@]}"; do
      gpu="${GPU_LIST[$i]}"
      echo "[eval] gpu=$gpu method=$method target=$target dataset=$dataset"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_agent_eval.py \
        --dataset "$dataset" --backend spec \
        --target "$MODEL_ROOT/$target" \
        --draft "$DRAFT_ROOT/$draft" \
        --output-dir "$out" \
        --num-gpus 1 --max-samples "$MAX_SAMPLES" \
        --max-steps "$MAX_STEPS" --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        > "$out/$dataset.log" 2>&1 &
      pids+=("$!")
      i=$((i + 1))
    done
  done
done

(( ${#pids[@]} > 0 )) || {
  echo "[eval] No tasks were launched" >&2
  exit 1
}

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || {
  echo "[eval] One or more tasks failed; inspect the logs under $OUT_ROOT" >&2
  exit 1
}

echo "[eval] Completed ${#pids[@]} tasks. Results are under $OUT_ROOT"
