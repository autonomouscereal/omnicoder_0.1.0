#!/usr/bin/env bash
set -euo pipefail

# Canonical AI-server launcher for the 20B-class native-1M target lane.
# Host GPUs 0,4,6 are exposed as container CUDA devices 0,1,2:
#   rank 0 -> RTX 3090, 16 layers
#   rank 1 -> RTX 3090, 16 layers
#   rank 2 -> RTX 8000, 32 layers plus final norm/head

REPO="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
RUN_TAG="${OMNICODER_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER_NAME="${OMNICODER_CONTAINER_NAME:-omnicoder_target20b_fast_${RUN_TAG}}"
OUT_DIR="${OMNICODER_OUT_DIR:-weights/training_orchestration_2026/target20b_fast_${RUN_TAG}}"
PROFILE="${OMNICODER_PROFILE:-profiles/training_orchestration_2026.json}"
MODE="${OMNICODER_MODE:-run-full}"

FAST_GPU_DEVICES="${OMNICODER_FAST_GPU_DEVICES:-0,4,6}"
RANK_DEVICE_MAP="${OMNICODER_RANK_DEVICE_MAP:-0,1,2}"
PLACEMENT_LAYER_COUNTS="${OMNICODER_PLACEMENT_LAYER_COUNTS:-16,16,32}"

START_STAGE="${OMNICODER_START_STAGE:-text}"
STAGE_ORDER="${OMNICODER_STAGE_ORDER:-text,code,tool,image,video,audio,music,long_context}"
RESUME_CHECKPOINT="${OMNICODER_RESUME_CHECKPOINT:-}"
POSTTRAIN_START_ALGORITHM="${OMNICODER_POSTTRAIN_START_ALGORITHM:-}"
POSTTRAIN_ALGORITHM_ORDER="${OMNICODER_POSTTRAIN_ALGORITHM_ORDER:-}"

STEPS_PER_STAGE="${OMNICODER_STEPS_PER_STAGE:-64}"
SEQ_LEN="${OMNICODER_SEQ_LEN:-1024}"
BATCH_SIZE="${OMNICODER_BATCH_SIZE:-1}"
LEARNING_RATE="${OMNICODER_LR:-0.00002}"
SAVE_INTERVAL="${OMNICODER_SAVE_INTERVAL:-32}"
POSTTRAIN_STEPS="${OMNICODER_POSTTRAIN_STEPS:-32}"
FINETUNE_STEPS="${OMNICODER_FINETUNE_STEPS:-64}"
DETACH="${OMNICODER_DETACH:-1}"
ADAPTIVE_WEIGHTS="${OMNICODER_ADAPTIVE_WEIGHTS:-1}"
MIXTURE_PLAN="${OMNICODER_MIXTURE_PLAN:-weights/training_orchestration_2026/manifests/mixture_plan.json}"
CONTEXT_LADDER="${OMNICODER_CONTEXT_LADDER:-8192,32768,131072,262144,524288,1048576}"
RLVR_ALGOS="${OMNICODER_RLVR_ALGOS:-grpo,dapo,offline_reward_replay}"
LIVE_POSTTRAIN="${OMNICODER_LIVE_POSTTRAIN:-0}"
POSTTRAIN_LR="${OMNICODER_POSTTRAIN_LR:-0}"
POSTTRAIN_MAX_RECORDS="${OMNICODER_POSTTRAIN_MAX_RECORDS:-0}"
FINETUNE_LR="${OMNICODER_FINETUNE_LR:-0}"
DISTILL_PROFILE="${OMNICODER_DISTILL_PROFILE:-}"
DISTILL_LIMIT="${OMNICODER_DISTILL_LIMIT:-0}"
DISTILL_STEPS="${OMNICODER_DISTILL_STEPS:-0}"
DISTILL_LR="${OMNICODER_DISTILL_LR:-0}"
BENCHMARK_SEQ_LEN="${OMNICODER_BENCHMARK_SEQ_LEN:-0}"
HELDOUT_MAX_RECORDS_PER_FILE="${OMNICODER_HELDOUT_MAX_RECORDS_PER_FILE:-}"
BENCHMARK_MAX_RECORDS_PER_FILE="${OMNICODER_BENCHMARK_MAX_RECORDS_PER_FILE:-}"
HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS="${OMNICODER_HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS:-0}"
BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS="${OMNICODER_BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS:-0}"
BENCHMARK_CYCLE="${OMNICODER_BENCHMARK_CYCLE:-}"
BENCHMARK_MIN_TASKS="${OMNICODER_BENCHMARK_MIN_TASKS:-0}"
BENCHMARK_PREDICTIONS="${OMNICODER_BENCHMARK_PREDICTIONS:-}"
REQUIRE_REPORTABLE_GATE="${OMNICODER_REQUIRE_REPORTABLE_GATE:-0}"
RERUN_HELDOUT_EVALS="${OMNICODER_RERUN_HELDOUT_EVALS:-0}"

cd "$REPO"
if [[ "$OUT_DIR" == /workspace/weights/* ]]; then
  HOST_OUT_DIR="$WEIGHTS_ROOT/${OUT_DIR#/workspace/weights/}"
elif [[ "$OUT_DIR" == weights/* ]]; then
  HOST_OUT_DIR="$WEIGHTS_ROOT/${OUT_DIR#weights/}"
elif [[ "$OUT_DIR" == /* ]]; then
  HOST_OUT_DIR="$OUT_DIR"
else
  HOST_OUT_DIR="$REPO/$OUT_DIR"
fi
mkdir -p "$HOST_OUT_DIR"

resume_args=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  resume_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

append_nonzero_arg() {
  local -n target_array="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "$value" && "$value" != "0" && "$value" != "0.0" ]]; then
    target_array+=("$flag" "$value")
  fi
}

append_nonempty_arg() {
  local -n target_array="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    target_array+=("$flag" "$value")
  fi
}

shared_eval_args=()
append_nonempty_arg shared_eval_args --heldout-max-records-per-file "$HELDOUT_MAX_RECORDS_PER_FILE"
append_nonempty_arg shared_eval_args --benchmark-max-records-per-file "$BENCHMARK_MAX_RECORDS_PER_FILE"
append_nonzero_arg shared_eval_args --heldout-sample-loss-timeout-seconds "$HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS"
append_nonzero_arg shared_eval_args --benchmark-sample-loss-timeout-seconds "$BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS"
append_nonempty_arg shared_eval_args --benchmark-cycle "$BENCHMARK_CYCLE"
append_nonzero_arg shared_eval_args --benchmark-min-tasks "$BENCHMARK_MIN_TASKS"
append_nonempty_arg shared_eval_args --benchmark-predictions "$BENCHMARK_PREDICTIONS"
if truthy "$REQUIRE_REPORTABLE_GATE"; then
  shared_eval_args+=(--require-reportable-gate)
fi
if truthy "$RERUN_HELDOUT_EVALS"; then
  shared_eval_args+=(--rerun-heldout-evals)
fi

shared_posttrain_args=()
append_nonzero_arg shared_posttrain_args --posttrain-lr "$POSTTRAIN_LR"
append_nonzero_arg shared_posttrain_args --posttrain-max-records "$POSTTRAIN_MAX_RECORDS"

full_only_args=()
append_nonempty_arg full_only_args --distill-profile "$DISTILL_PROFILE"
append_nonzero_arg full_only_args --distill-limit "$DISTILL_LIMIT"
append_nonzero_arg full_only_args --distill-steps "$DISTILL_STEPS"
append_nonzero_arg full_only_args --distill-lr "$DISTILL_LR"
append_nonzero_arg full_only_args --finetune-lr "$FINETUNE_LR"
append_nonzero_arg full_only_args --benchmark-seq-len "$BENCHMARK_SEQ_LEN"

if [[ "$MODE" == "run-posttraining" || "$MODE" == "run-posttrain" ]]; then
  if [[ -z "$RESUME_CHECKPOINT" ]]; then
    echo "OMNICODER_RESUME_CHECKPOINT is required for $MODE" >&2
    exit 2
  fi
  posttrain_start_args=()
  if [[ -n "$POSTTRAIN_START_ALGORITHM" ]]; then
    posttrain_start_args+=(--posttrain-start-algorithm "$POSTTRAIN_START_ALGORITHM")
  fi
  posttrain_order_args=()
  if [[ -n "$POSTTRAIN_ALGORITHM_ORDER" ]]; then
    posttrain_order_args+=(--posttrain-algorithm-order "$POSTTRAIN_ALGORITHM_ORDER")
  fi
  common_args=(
    --profile "$PROFILE"
    --out-dir "$OUT_DIR"
    "$MODE"
    --preset omnicoder2026_20b_1m
    --resume-checkpoint "$RESUME_CHECKPOINT"
    "${posttrain_start_args[@]}"
    "${posttrain_order_args[@]}"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --posttrain-steps "$POSTTRAIN_STEPS"
    --distributed pipeline_stage
    --nproc-per-node 3
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --pipeline-stage-schedule gpipe
    --pipeline-microbatches 1
    --precision fp16
    --init-dtype fp16
    --optimizer adafactor
    --optimizer-in-backward
    --optimizer-in-backward-update lowmem_adafactor
    --optimizer-in-backward-grad-clip 1.0
    --optimizer-in-backward-clip-mode rms
    --optimizer-in-backward-adafactor-chunk-rows 256
    --optimizer-in-backward-adafactor-clip-threshold 1.0
    --optimizer-in-backward-adafactor-decay-rate -0.8
    --optimizer-in-backward-adafactor-eps1 1e-30
    --activation-checkpointing
    --fake-quant-chunk-rows 64
    --fake-quant-max-full-elements 16777216
    "${shared_posttrain_args[@]}"
    "${shared_eval_args[@]}"
    --fake-quant
  )
else
  common_args=(
    --profile "$PROFILE"
    --out-dir "$OUT_DIR"
    "$MODE"
    --preset omnicoder2026_20b_1m
    "${resume_args[@]}"
    --start-stage "$START_STAGE"
    --stage-order "$STAGE_ORDER"
    --steps-per-stage "$STEPS_PER_STAGE"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --lr "$LEARNING_RATE"
    --save-interval "$SAVE_INTERVAL"
    --posttrain-steps "$POSTTRAIN_STEPS"
    --distributed pipeline_stage
    --nproc-per-node 3
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --pipeline-stage-schedule gpipe
    --pipeline-microbatches 1
    --precision fp16
    --init-dtype fp16
    --optimizer adafactor
    --optimizer-in-backward
    --optimizer-in-backward-update lowmem_adafactor
    --optimizer-in-backward-grad-clip 1.0
    --optimizer-in-backward-clip-mode rms
    --optimizer-in-backward-adafactor-chunk-rows 256
    --optimizer-in-backward-adafactor-clip-threshold 1.0
    --optimizer-in-backward-adafactor-decay-rate -0.8
    --optimizer-in-backward-adafactor-eps1 1e-30
    --activation-checkpointing
    --fake-quant-chunk-rows 64
    --fake-quant-max-full-elements 16777216
    "${shared_posttrain_args[@]}"
    "${shared_eval_args[@]}"
    --fake-quant
  )
  if [[ "$MODE" == "run-full" ]]; then
    common_args+=(--finetune-steps "$FINETUNE_STEPS")
    common_args+=("${full_only_args[@]}")
  elif truthy "$LIVE_POSTTRAIN"; then
    common_args+=(--live-posttraining)
  fi
fi

docker_args=(
  --name "$CONTAINER_NAME"
  --gpus "\"device=${FAST_GPU_DEVICES}\""
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e PYTHONPATH=/workspace/src
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  -e NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
  -e TOKENIZERS_PARALLELISM=false
  -e OMNICODER_ADAPTIVE_WEIGHTS="$ADAPTIVE_WEIGHTS"
  -e OMNICODER_MIXTURE_PLAN="$MIXTURE_PLAN"
  -e OMNICODER_CONTEXT_LADDER="$CONTEXT_LADDER"
  -e OMNICODER_RLVR_ALGOS="$RLVR_ALGOS"
  -v "$REPO:/workspace"
  -v "$WEIGHTS_ROOT:/workspace/weights"
  -v /home/cereal:/home/cereal:ro
  -w /workspace
)

run_cmd=(
  bash -lc
  "set -euo pipefail; python -m omnicoder.training.training_orchestration_2026 ${common_args[*]}"
)

if [[ "$DETACH" == "1" ]]; then
  docker run -d "${docker_args[@]}" "$IMAGE" "${run_cmd[@]}"
  echo "container=$CONTAINER_NAME"
  echo "out_dir=$OUT_DIR"
  echo "host_out_dir=$HOST_OUT_DIR"
  echo "logs: docker logs -f $CONTAINER_NAME"
else
  docker run --rm "${docker_args[@]}" "$IMAGE" "${run_cmd[@]}"
fi
