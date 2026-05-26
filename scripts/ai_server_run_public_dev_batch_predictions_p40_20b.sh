#!/usr/bin/env bash
set -euo pipefail

# Persistent public-dev prediction runner for a complete 3-shard Omnicoder 20B
# pipeline checkpoint on the AI-server P40 layout. This produces local
# engineering prediction artifacts; official/reportable status is decided by the
# benchmark suite scorer and the dataset metadata.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
RUN_TAG_RAW="${OMNICODER_BATCH_PRED_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
RECOVERY_RUN="${OMNICODER_BATCH_PRED_RECOVERY_RUN:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_recovergrpo_d28a1d4_20260526T051052Z}"
OUT_DIR="${OMNICODER_BATCH_PRED_OUT_DIR:-weights/benchmarks_2026/public_dev_batch_predictions_${RUN_TAG}}"
HOST_OUT_DIR="$WEIGHTS_ROOT/${OUT_DIR#weights/}"
LOG_DIR="$HOST_OUT_DIR/logs"

GPU_DEVICES="${OMNICODER_BATCH_PRED_GPU_DEVICES:-${OMNICODER_FAST_GPU_DEVICES:-0,4,6}}"
NPROC_PER_NODE="${OMNICODER_BATCH_PRED_NPROC_PER_NODE:-3}"
RANK_DEVICE_MAP="${OMNICODER_BATCH_PRED_RANK_DEVICE_MAP:-0,1,2}"
PLACEMENT_LAYER_COUNTS="${OMNICODER_BATCH_PRED_PLACEMENT_LAYER_COUNTS:-16,16,32}"
PRECISION="${OMNICODER_BATCH_PRED_PRECISION:-fp16}"
INIT_DTYPE="${OMNICODER_BATCH_PRED_INIT_DTYPE:-fp16}"
PRESET="${OMNICODER_BATCH_PRED_PRESET:-omnicoder2026_20b_1m}"
PUBLIC_DEV_TASK_ROOTS="${OMNICODER_BATCH_PRED_TASK_ROOTS:-$WEIGHTS_ROOT/official_benchmarks_2026/runs/bench_reportable_fix_eaa2463_20260525T181734Z/local_2026}"
MAX_OUTPUT_TOKENS="${OMNICODER_BATCH_PRED_MAX_OUTPUT_TOKENS:-256}"
MAX_PROMPT_TOKENS="${OMNICODER_BATCH_PRED_MAX_PROMPT_TOKENS:-4096}"
PROGRESS_TASKS="${OMNICODER_BATCH_PRED_PROGRESS_TASKS:-1}"
FAKE_QUANT="${OMNICODER_BATCH_PRED_FAKE_QUANT:-1}"
FAKE_QUANT_CHUNK_ROWS="${OMNICODER_BATCH_PRED_FAKE_QUANT_CHUNK_ROWS:-16}"
FAKE_QUANT_MAX_FULL_ELEMENTS="${OMNICODER_BATCH_PRED_FAKE_QUANT_MAX_FULL_ELEMENTS:-16777216}"
DIST_TIMEOUT_SECONDS="${OMNICODER_BATCH_PRED_DIST_TIMEOUT_SECONDS:-7200}"
CUDA_ALLOC_CONF="${OMNICODER_BATCH_PRED_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
PYTHON_BIN="${OMNICODER_BATCH_PRED_PYTHON:-python}"

mkdir -p "$HOST_OUT_DIR" "$LOG_DIR"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

host_path() {
  local value="$1"
  if [[ "$value" == /workspace/weights/* ]]; then
    printf '%s\n' "$WEIGHTS_ROOT/${value#/workspace/weights/}"
  elif [[ "$value" == /workspace/* ]]; then
    printf '%s\n' "$REPO/${value#/workspace/}"
  elif [[ "$value" == weights/* ]]; then
    printf '%s\n' "$WEIGHTS_ROOT/${value#weights/}"
  elif [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$REPO/$value"
  fi
}

container_path() {
  local value="$1"
  if [[ "$value" == "$WEIGHTS_ROOT"/* ]]; then
    printf '%s\n' "/workspace/weights/${value#"$WEIGHTS_ROOT"/}"
  elif [[ "$value" == "$REPO"/* ]]; then
    printf '%s\n' "/workspace/${value#"$REPO"/}"
  elif [[ "$value" == weights/* ]]; then
    printf '%s\n' "/workspace/$value"
  elif [[ "$value" == /workspace/* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$value"
  fi
}

checkpoint_is_complete_3shard() {
  local checkpoint="$1"
  [[ -d "$checkpoint" ]] || return 1
  [[ -s "$checkpoint/manifest.json" ]] || return 1
  [[ -s "$checkpoint/.complete.json" ]] || return 1
  grep -Eq '"world_size"[[:space:]]*:[[:space:]]*3' "$checkpoint/manifest.json" || return 1
  local rank rank_file marker count
  for rank in 0 1 2; do
    printf -v rank_file 'rank%05d.pt' "$rank"
    marker="$checkpoint/${rank_file}.complete.json"
    [[ -s "$checkpoint/$rank_file" ]] || return 1
    [[ -s "$marker" ]] || return 1
  done
  count="$(find "$checkpoint" -maxdepth 1 -type f -name 'rank*.pt' | wc -l)"
  [[ "$count" -eq 3 ]] || return 1
}

find_latest_checkpoint() {
  local root="$RECOVERY_RUN/checkpoints/posttrain"
  local best=""
  local best_mtime=0
  local candidate mtime
  [[ -d "$root" ]] || return 1
  shopt -s nullglob
  for candidate in "$root"/*; do
    if checkpoint_is_complete_3shard "$candidate"; then
      mtime="$(stat -c '%Y' "$candidate/.complete.json")"
      if [[ "$mtime" -ge "$best_mtime" ]]; then
        best="$candidate"
        best_mtime="$mtime"
      fi
    fi
  done
  shopt -u nullglob
  [[ -n "$best" ]] || return 1
  printf '%s\n' "$best"
}

CHECKPOINT_INPUT="${OMNICODER_BATCH_PRED_CHECKPOINT:-}"
if [[ -z "$CHECKPOINT_INPUT" ]]; then
  CHECKPOINT_INPUT="$(find_latest_checkpoint)" || {
    echo "No complete 3-shard checkpoint found under $RECOVERY_RUN/checkpoints/posttrain." >&2
    exit 4
  }
fi
CHECKPOINT_HOST="$(host_path "$CHECKPOINT_INPUT")"
CHECKPOINT_CONTAINER="$(container_path "$CHECKPOINT_HOST")"

if ! checkpoint_is_complete_3shard "$CHECKPOINT_HOST"; then
  echo "Checkpoint is not a complete 3-shard Omnicoder pipeline checkpoint: $CHECKPOINT_HOST" >&2
  exit 4
fi

public_dev_roots_present=()
IFS=',' read -r -a public_roots <<< "$PUBLIC_DEV_TASK_ROOTS"
for root in "${public_roots[@]}"; do
  root="${root#"${root%%[![:space:]]*}"}"
  root="${root%"${root##*[![:space:]]}"}"
  [[ -n "$root" ]] || continue
  root_host="$(host_path "$root")"
  if [[ -d "$root_host" ]] && find "$root_host" -type f -name '*.jsonl' -print -quit | grep -q .; then
    public_dev_roots_present+=("$root_host")
  elif [[ -f "$root_host" && "$root_host" == *.jsonl ]]; then
    public_dev_roots_present+=("$root_host")
  fi
done

if [[ "${#public_dev_roots_present[@]}" -eq 0 ]]; then
  echo "No public-dev task JSONL files found. Set OMNICODER_BATCH_PRED_TASK_ROOTS." >&2
  exit 6
fi

task_args=()
for root in "${public_dev_roots_present[@]}"; do
  task_args+=(--tasks "$(container_path "$root")")
done

cmd=(
  "$PYTHON_BIN" -m omnicoder.eval.pipeline_checkpoint_batch_predict_2026
  --checkpoint "$CHECKPOINT_CONTAINER"
  "${task_args[@]}"
  --out "/workspace/$OUT_DIR/public_dev_predictions.batch_p40_20b.jsonl"
  --summary "/workspace/$OUT_DIR/public_dev_prediction_summary.batch_p40_20b.json"
  --model "$CHECKPOINT_CONTAINER"
  --preset "$PRESET"
  --nproc-per-node "$NPROC_PER_NODE"
  --rank-device-map "$RANK_DEVICE_MAP"
  --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
  --precision "$PRECISION"
  --init-dtype "$INIT_DTYPE"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --max-prompt-tokens "$MAX_PROMPT_TOKENS"
  --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS"
  --progress-tasks "$PROGRESS_TASKS"
  --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
  --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
  --require-target-contract
  --allow-p40-target-contract-eval
  --force
)

if truthy "$FAKE_QUANT"; then
  cmd+=(--fake-quant)
fi

printf -v quoted '%q ' "${cmd[@]}"
{
  printf 'checkpoint=%s\n' "$CHECKPOINT_HOST"
  printf 'checkpoint_container=%s\n' "$CHECKPOINT_CONTAINER"
  printf 'out_dir=%s\n' "$HOST_OUT_DIR"
  printf 'task_roots:\n'
  printf '  %s\n' "${public_dev_roots_present[@]}"
  printf 'command=%s\n' "$quoted"
} > "$HOST_OUT_DIR/input_files.batch_p40_20b.txt"

docker run \
  --name "omnicoder_public_dev_batch_predictions_${RUN_TAG}" \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONPATH=/workspace/src \
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  -e NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}" \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF" \
  -e OMNICODER2026_DIST_TIMEOUT_SECONDS="$DIST_TIMEOUT_SECONDS" \
  -e OMNICODER2026_FAKE_QUANT_CHUNK_ROWS="$FAKE_QUANT_CHUNK_ROWS" \
  -e OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS="$FAKE_QUANT_MAX_FULL_ELEMENTS" \
  -e OMNICODER2026_FFN_CHUNK_TOKENS="${OMNICODER_BATCH_PRED_FFN_CHUNK_TOKENS:-256}" \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$REPO:/workspace" \
  -v "$WEIGHTS_ROOT:/workspace/weights" \
  -v /home/cereal:/home/cereal:ro \
  -w /workspace \
  "$IMAGE" \
  bash -lc "set -euo pipefail; $quoted" 2>&1 | tee "$LOG_DIR/public_dev_batch_predictions.log"

echo "public_dev_batch_predictions_out=$HOST_OUT_DIR/public_dev_predictions.batch_p40_20b.jsonl"
echo "public_dev_batch_predictions_summary=$HOST_OUT_DIR/public_dev_prediction_summary.batch_p40_20b.json"
