#!/usr/bin/env bash
set -euo pipefail

# Canonical AI-server launcher for the 20B-class native-1M target lane.
# Host GPUs 0,4,6 are exposed as container CUDA devices 0,1,2:
#   rank 0 -> RTX 3090, 16 layers
#   rank 1 -> RTX 3090, 16 layers
#   rank 2 -> RTX 8000, 32 layers plus final norm/head

REPO="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
AI_DATA_ROOT="${OMNICODER_AI_DATA_ROOT:-/mnt/ai_data}"
IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
RUN_TAG="${OMNICODER_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER_NAME="${OMNICODER_CONTAINER_NAME:-omnicoder_target20b_fast_${RUN_TAG}}"
OUT_DIR="${OMNICODER_OUT_DIR:-weights/training_orchestration_2026/target20b_fast_${RUN_TAG}}"
PROFILE="${OMNICODER_PROFILE:-profiles/training_orchestration_2026.json}"
MODE="${OMNICODER_MODE:-run-full}"

FAST_GPU_DEVICES="${OMNICODER_FAST_GPU_DEVICES:-0,4,6}"
RANK_DEVICE_MAP="${OMNICODER_RANK_DEVICE_MAP:-0,1,2}"
PLACEMENT_LAYER_COUNTS="${OMNICODER_PLACEMENT_LAYER_COUNTS:-16,16,32}"
IFS=',' read -r -a RANK_DEVICE_MAP_ITEMS <<< "$RANK_DEVICE_MAP"
NPROC_PER_NODE="${OMNICODER_NPROC_PER_NODE:-${#RANK_DEVICE_MAP_ITEMS[@]}}"
CUDA_ALLOC_CONF="${OMNICODER_CUDA_ALLOC_CONF:-expandable_segments:True}"

START_STAGE="${OMNICODER_START_STAGE:-text}"
STAGE_ORDER="${OMNICODER_STAGE_ORDER:-text,code,tool,image,video,audio,music,tts,ocr,long_context}"
RESUME_CHECKPOINT="${OMNICODER_RESUME_CHECKPOINT:-}"
CURATION_MANIFEST="${OMNICODER_CURATION_MANIFEST:-}"
POSTTRAIN_START_ALGORITHM="${OMNICODER_POSTTRAIN_START_ALGORITHM:-}"
POSTTRAIN_ALGORITHM_ORDER="${OMNICODER_POSTTRAIN_ALGORITHM_ORDER:-}"
POSTTRAIN_INPUT_JSONL="${OMNICODER_POSTTRAIN_INPUT_JSONL:-}"

STEPS_PER_STAGE="${OMNICODER_STEPS_PER_STAGE:-64}"
SEQ_LEN="${OMNICODER_SEQ_LEN:-1024}"
BATCH_SIZE="${OMNICODER_BATCH_SIZE:-1}"
LEARNING_RATE="${OMNICODER_LR:-0.00002}"
# Set OMNICODER_SAVE_INTERVAL=0 for no interval checkpoints. Set
# OMNICODER2026_SKIP_FINAL_SAVE=1 only for bounded profiling runs where no
# checkpoint should be written at all.
SAVE_INTERVAL="${OMNICODER_SAVE_INTERVAL:-32}"
SKIP_FINAL_SAVE="${OMNICODER2026_SKIP_FINAL_SAVE:-0}"
FAKE_QUANT="${OMNICODER_FAKE_QUANT:-1}"
FAKE_QUANT_CHUNK_ROWS="${OMNICODER_FAKE_QUANT_CHUNK_ROWS:-256}"
FAKE_QUANT_MAX_FULL_ELEMENTS="${OMNICODER_FAKE_QUANT_MAX_FULL_ELEMENTS:-16777216}"
ACTIVATION_CHECKPOINTING="${OMNICODER_ACTIVATION_CHECKPOINTING:-1}"
PIPELINE_STAGE_SCHEDULE="${OMNICODER_PIPELINE_STAGE_SCHEDULE:-${OMNICODER_PIPELINE_SCHEDULE:-gpipe}}"
PIPELINE_MICROBATCHES="${OMNICODER_PIPELINE_MICROBATCHES:-1}"
LM_LOSS_CHUNK_TOKENS="${OMNICODER_LM_LOSS_CHUNK_TOKENS:-64}"
FFN_CHUNK_TOKENS="${OMNICODER_FFN_CHUNK_TOKENS:-256}"
LOSS_TOKEN_STRIDE="${OMNICODER_LOSS_TOKEN_STRIDE:-1}"
MAX_LOSS_TOKENS_PER_SAMPLE="${OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE:-64}"
OPTIMIZER_ADAFACTOR_CHUNK_ROWS="${OMNICODER_OPTIMIZER_IN_BACKWARD_ADAFACTOR_CHUNK_ROWS:-256}"
OPTIMIZER_IN_BACKWARD="${OMNICODER_OPTIMIZER_IN_BACKWARD:-1}"
DIST_TIMEOUT_SECONDS="${OMNICODER2026_DIST_TIMEOUT_SECONDS:-7200}"
CHECKPOINT_SYNC_BACKEND="${OMNICODER2026_CHECKPOINT_SYNC_BACKEND:-filesystem}"
CHECKPOINT_MARKER_TIMEOUT_SECONDS="${OMNICODER2026_CHECKPOINT_MARKER_TIMEOUT_SECONDS:-14400}"
CHECKPOINT_MARKER_POLL_SECONDS="${OMNICODER2026_CHECKPOINT_MARKER_POLL_SECONDS:-2}"
STEP_TIMING_INTERVAL="${OMNICODER2026_STEP_TIMING_INTERVAL:-8}"
RANK_SKEW_INTERVAL="${OMNICODER2026_RANK_SKEW_INTERVAL:-32}"
LOSS_DIAGNOSTICS_INTERVAL="${OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL:-8}"
DETAILED_EVENT_LOG_INTERVAL="${OMNICODER2026_DETAILED_EVENT_LOG_INTERVAL:-0}"
TIMING_CUDA_SYNC="${OMNICODER2026_TIMING_CUDA_SYNC:-0}"
BLOCK_TIMING="${OMNICODER2026_BLOCK_TIMING:-0}"
BLOCK_TIMING_CUDA_SYNC="${OMNICODER2026_BLOCK_TIMING_CUDA_SYNC:-0}"
CHECKPOINT_DATA_HASH_POLICY="${OMNICODER2026_CHECKPOINT_DATA_HASH_POLICY:-manifest}"
PIPELINE_REASONING_EFFORT="${OMNICODER2026_PIPELINE_REASONING_EFFORT:-0}"
POSTTRAIN_STEPS="${OMNICODER_POSTTRAIN_STEPS:-32}"
FINETUNE_STEPS="${OMNICODER_FINETUNE_STEPS:-64}"
DETACH="${OMNICODER_DETACH:-1}"
ADAPTIVE_WEIGHTS="${OMNICODER_ADAPTIVE_WEIGHTS:-1}"
MIXTURE_PLAN="${OMNICODER_MIXTURE_PLAN:-weights/training_orchestration_2026/manifests/mixture_plan.json}"
CONTEXT_LADDER="${OMNICODER_CONTEXT_LADDER:-8192,32768,131072,262144,524288,1048576}"
LONG_CONTEXT_STEPS_PER_RUNG="${OMNICODER_LONG_CONTEXT_STEPS_PER_RUNG:-0}"
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
BENCHMARK_PREDICTION_BACKEND="${OMNICODER_BENCHMARK_PREDICTION_BACKEND:-}"
BENCHMARK_PREDICTION_MODEL="${OMNICODER_BENCHMARK_PREDICTION_MODEL:-}"
BENCHMARK_PREDICTION_BASE_URL="${OMNICODER_BENCHMARK_PREDICTION_BASE_URL:-}"
BENCHMARK_PREDICTION_API_KEY_ENV="${OMNICODER_BENCHMARK_PREDICTION_API_KEY_ENV:-}"
BENCHMARK_PREDICTION_CHECKPOINT_RUNNER="${OMNICODER_BENCHMARK_PREDICTION_CHECKPOINT_RUNNER:-}"
BENCHMARK_PREDICTION_TIMEOUT_SECONDS="${OMNICODER_BENCHMARK_PREDICTION_TIMEOUT_SECONDS:-0}"
BENCHMARK_PREDICTION_MAX_OUTPUT_TOKENS="${OMNICODER_BENCHMARK_PREDICTION_MAX_OUTPUT_TOKENS:-0}"
REPORTABLE_TASK_ROOTS="${OMNICODER_REPORTABLE_TASK_ROOTS:-}"
REQUIRE_REPORTABLE_GATE="${OMNICODER_REQUIRE_REPORTABLE_GATE:-0}"
RERUN_HELDOUT_EVALS="${OMNICODER_RERUN_HELDOUT_EVALS:-0}"
BENCHMARK_MATERIALIZATION_ROOT="${OMNICODER_BENCHMARK_MATERIALIZATION_ROOT:-}"
ALLOW_LOCAL_BENCHMARK_TASK_ROOTS="${OMNICODER_ALLOW_LOCAL_BENCHMARK_TASK_ROOTS:-0}"
CHECKPOINT_READINESS_REPORT="${OMNICODER_CHECKPOINT_READINESS_REPORT:-}"
CHECKPOINT_TOPK_PROBE="${OMNICODER_CHECKPOINT_TOPK_PROBE:-}"
CHECKPOINT_SAMPLE_LOSS="${OMNICODER_CHECKPOINT_SAMPLE_LOSS:-}"
CHECKPOINT_MEDIA_ROUTE_PROBE="${OMNICODER_CHECKPOINT_MEDIA_ROUTE_PROBE:-}"
CHECKPOINT_READINESS_MAX_AVG_LOSS="${OMNICODER_CHECKPOINT_READINESS_MAX_AVG_LOSS:-20}"
CHECKPOINT_READINESS_MAX_PERPLEXITY="${OMNICODER_CHECKPOINT_READINESS_MAX_PERPLEXITY:-1000000}"
CHECKPOINT_READINESS_MIN_TOKENS="${OMNICODER_CHECKPOINT_READINESS_MIN_TOKENS:-64}"
CHECKPOINT_READINESS_MIN_WEIGHT_STD="${OMNICODER_CHECKPOINT_READINESS_MIN_WEIGHT_STD:-0.00001}"
CHECKPOINT_READINESS_MAX_WEIGHT_STD="${OMNICODER_CHECKPOINT_READINESS_MAX_WEIGHT_STD:-0.2}"
AUTO_CHECKPOINT_MEDIA_ROUTE_PROBE="${OMNICODER_AUTO_CHECKPOINT_MEDIA_ROUTE_PROBE:-1}"

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

optimizer_in_backward_args=()
if truthy "$OPTIMIZER_IN_BACKWARD"; then
  optimizer_in_backward_args+=(--optimizer-in-backward)
fi

fake_quant_args=()
if truthy "$FAKE_QUANT"; then
  fake_quant_args+=(--fake-quant)
fi

activation_checkpointing_args=()
if truthy "$ACTIVATION_CHECKPOINTING"; then
  activation_checkpointing_args+=(--activation-checkpointing)
fi

if [[ -n "$RESUME_CHECKPOINT" && -z "$CHECKPOINT_READINESS_REPORT" && -z "$CHECKPOINT_MEDIA_ROUTE_PROBE" ]] && truthy "$AUTO_CHECKPOINT_MEDIA_ROUTE_PROBE"; then
  CHECKPOINT_MEDIA_ROUTE_PROBE="$OUT_DIR/readiness/media_route_probe.json"
fi

shared_checkpoint_readiness_args=()
append_nonempty_arg shared_checkpoint_readiness_args --checkpoint-readiness-report "$CHECKPOINT_READINESS_REPORT"
append_nonempty_arg shared_checkpoint_readiness_args --checkpoint-topk-probe "$CHECKPOINT_TOPK_PROBE"
append_nonempty_arg shared_checkpoint_readiness_args --checkpoint-sample-loss "$CHECKPOINT_SAMPLE_LOSS"
append_nonempty_arg shared_checkpoint_readiness_args --checkpoint-media-route-probe "$CHECKPOINT_MEDIA_ROUTE_PROBE"
append_nonzero_arg shared_checkpoint_readiness_args --checkpoint-readiness-max-avg-loss "$CHECKPOINT_READINESS_MAX_AVG_LOSS"
append_nonzero_arg shared_checkpoint_readiness_args --checkpoint-readiness-max-perplexity "$CHECKPOINT_READINESS_MAX_PERPLEXITY"
append_nonzero_arg shared_checkpoint_readiness_args --checkpoint-readiness-min-tokens "$CHECKPOINT_READINESS_MIN_TOKENS"
append_nonzero_arg shared_checkpoint_readiness_args --checkpoint-readiness-min-weight-std "$CHECKPOINT_READINESS_MIN_WEIGHT_STD"
append_nonzero_arg shared_checkpoint_readiness_args --checkpoint-readiness-max-weight-std "$CHECKPOINT_READINESS_MAX_WEIGHT_STD"

shared_eval_args=()
append_nonempty_arg shared_eval_args --heldout-max-records-per-file "$HELDOUT_MAX_RECORDS_PER_FILE"
append_nonempty_arg shared_eval_args --benchmark-max-records-per-file "$BENCHMARK_MAX_RECORDS_PER_FILE"
append_nonzero_arg shared_eval_args --heldout-sample-loss-timeout-seconds "$HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS"
append_nonzero_arg shared_eval_args --benchmark-sample-loss-timeout-seconds "$BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS"
append_nonempty_arg shared_eval_args --benchmark-cycle "$BENCHMARK_CYCLE"
append_nonzero_arg shared_eval_args --benchmark-min-tasks "$BENCHMARK_MIN_TASKS"
append_nonempty_arg shared_eval_args --benchmark-predictions "$BENCHMARK_PREDICTIONS"
append_nonempty_arg shared_eval_args --benchmark-prediction-backend "$BENCHMARK_PREDICTION_BACKEND"
append_nonempty_arg shared_eval_args --benchmark-prediction-model "$BENCHMARK_PREDICTION_MODEL"
append_nonempty_arg shared_eval_args --benchmark-prediction-base-url "$BENCHMARK_PREDICTION_BASE_URL"
append_nonempty_arg shared_eval_args --benchmark-prediction-api-key-env "$BENCHMARK_PREDICTION_API_KEY_ENV"
append_nonempty_arg shared_eval_args --benchmark-prediction-checkpoint-runner "$BENCHMARK_PREDICTION_CHECKPOINT_RUNNER"
append_nonzero_arg shared_eval_args --benchmark-prediction-timeout-seconds "$BENCHMARK_PREDICTION_TIMEOUT_SECONDS"
append_nonzero_arg shared_eval_args --benchmark-prediction-max-output-tokens "$BENCHMARK_PREDICTION_MAX_OUTPUT_TOKENS"

append_reportable_task_root_if_jsonl() {
  local candidate="$1"
  if [[ -d "$candidate" ]] && compgen -G "$candidate/*.jsonl" >/dev/null; then
    shared_eval_args+=(--reportable-task-root "$candidate")
  fi
}

if [[ -n "$REPORTABLE_TASK_ROOTS" ]]; then
  IFS=',' read -r -a reportable_task_roots <<< "$REPORTABLE_TASK_ROOTS"
  for reportable_task_root in "${reportable_task_roots[@]}"; do
    if [[ -n "$reportable_task_root" ]]; then
      shared_eval_args+=(--reportable-task-root "$reportable_task_root")
    fi
  done
fi
if [[ -n "$BENCHMARK_MATERIALIZATION_ROOT" ]]; then
  append_reportable_task_root_if_jsonl "$BENCHMARK_MATERIALIZATION_ROOT/reportable_2026"
  if truthy "$ALLOW_LOCAL_BENCHMARK_TASK_ROOTS"; then
    append_reportable_task_root_if_jsonl "$BENCHMARK_MATERIALIZATION_ROOT/local_2026"
  fi
fi
if truthy "$REQUIRE_REPORTABLE_GATE"; then
  shared_eval_args+=(--require-reportable-gate)
fi
if truthy "$RERUN_HELDOUT_EVALS"; then
  shared_eval_args+=(--rerun-heldout-evals)
fi

shared_posttrain_args=()
append_nonzero_arg shared_posttrain_args --posttrain-lr "$POSTTRAIN_LR"
append_nonzero_arg shared_posttrain_args --posttrain-max-records "$POSTTRAIN_MAX_RECORDS"
if [[ -n "$POSTTRAIN_INPUT_JSONL" ]]; then
  IFS=',' read -r -a posttrain_input_jsonls <<< "$POSTTRAIN_INPUT_JSONL"
  for posttrain_input_jsonl in "${posttrain_input_jsonls[@]}"; do
    if [[ -n "$posttrain_input_jsonl" ]]; then
      shared_posttrain_args+=(--posttrain-input-jsonl "$posttrain_input_jsonl")
    fi
  done
fi

full_only_args=()
append_nonempty_arg full_only_args --distill-profile "$DISTILL_PROFILE"
append_nonzero_arg full_only_args --distill-limit "$DISTILL_LIMIT"
append_nonzero_arg full_only_args --distill-steps "$DISTILL_STEPS"
append_nonzero_arg full_only_args --distill-lr "$DISTILL_LR"
append_nonzero_arg full_only_args --finetune-lr "$FINETUNE_LR"
append_nonzero_arg full_only_args --benchmark-seq-len "$BENCHMARK_SEQ_LEN"

long_context_args=()
append_nonempty_arg long_context_args --context-ladder "$CONTEXT_LADDER"
append_nonzero_arg long_context_args --long-context-steps-per-rung "$LONG_CONTEXT_STEPS_PER_RUNG"

if [[ "$MODE" == "run-long-context" || "$MODE" == "run-longctx" ]]; then
  if [[ -z "$RESUME_CHECKPOINT" ]]; then
    echo "OMNICODER_RESUME_CHECKPOINT is required for $MODE" >&2
    exit 2
  fi
  curation_manifest_args=()
  append_nonempty_arg curation_manifest_args --curation-manifest "$CURATION_MANIFEST"
  common_args=(
    --profile "$PROFILE"
    --out-dir "$OUT_DIR"
    "$MODE"
    --preset omnicoder2026_20b_1m
    --resume-checkpoint "$RESUME_CHECKPOINT"
    "${curation_manifest_args[@]}"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --lr "$LEARNING_RATE"
    --save-interval "$SAVE_INTERVAL"
    "${long_context_args[@]}"
    --distributed pipeline_stage
    --nproc-per-node "$NPROC_PER_NODE"
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --pipeline-stage-schedule "$PIPELINE_STAGE_SCHEDULE"
    --pipeline-microbatches "$PIPELINE_MICROBATCHES"
    --precision fp16
    --init-dtype fp16
    --optimizer adafactor
    "${optimizer_in_backward_args[@]}"
    --optimizer-in-backward-update lowmem_adafactor
    --optimizer-in-backward-grad-clip 1.0
    --optimizer-in-backward-clip-mode rms
    --optimizer-in-backward-adafactor-chunk-rows "$OPTIMIZER_ADAFACTOR_CHUNK_ROWS"
    --optimizer-in-backward-adafactor-clip-threshold 1.0
    --optimizer-in-backward-adafactor-decay-rate -0.8
    --optimizer-in-backward-adafactor-eps1 1e-30
    "${activation_checkpointing_args[@]}"
    --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
    --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
    "${shared_eval_args[@]}"
    "${shared_checkpoint_readiness_args[@]}"
    "${fake_quant_args[@]}"
  )
elif [[ "$MODE" == "run-posttraining" || "$MODE" == "run-posttrain" ]]; then
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
  curation_manifest_args=()
  append_nonempty_arg curation_manifest_args --curation-manifest "$CURATION_MANIFEST"
  common_args=(
    --profile "$PROFILE"
    --out-dir "$OUT_DIR"
    "$MODE"
    --preset omnicoder2026_20b_1m
    --resume-checkpoint "$RESUME_CHECKPOINT"
    "${posttrain_start_args[@]}"
    "${posttrain_order_args[@]}"
    "${curation_manifest_args[@]}"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --posttrain-steps "$POSTTRAIN_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --distributed pipeline_stage
    --nproc-per-node "$NPROC_PER_NODE"
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --pipeline-stage-schedule "$PIPELINE_STAGE_SCHEDULE"
    --pipeline-microbatches "$PIPELINE_MICROBATCHES"
    --precision fp16
    --init-dtype fp16
    --optimizer adafactor
    "${optimizer_in_backward_args[@]}"
    --optimizer-in-backward-update lowmem_adafactor
    --optimizer-in-backward-grad-clip 1.0
    --optimizer-in-backward-clip-mode rms
    --optimizer-in-backward-adafactor-chunk-rows "$OPTIMIZER_ADAFACTOR_CHUNK_ROWS"
    --optimizer-in-backward-adafactor-clip-threshold 1.0
    --optimizer-in-backward-adafactor-decay-rate -0.8
    --optimizer-in-backward-adafactor-eps1 1e-30
    "${activation_checkpointing_args[@]}"
    --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
    --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
    "${shared_posttrain_args[@]}"
    "${shared_eval_args[@]}"
    "${shared_checkpoint_readiness_args[@]}"
    "${fake_quant_args[@]}"
  )
else
  curation_manifest_args=()
  append_nonempty_arg curation_manifest_args --curation-manifest "$CURATION_MANIFEST"
  common_args=(
    --profile "$PROFILE"
    --out-dir "$OUT_DIR"
    "$MODE"
    --preset omnicoder2026_20b_1m
    "${resume_args[@]}"
    "${curation_manifest_args[@]}"
    --start-stage "$START_STAGE"
    --stage-order "$STAGE_ORDER"
    --steps-per-stage "$STEPS_PER_STAGE"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --lr "$LEARNING_RATE"
    --save-interval "$SAVE_INTERVAL"
    --posttrain-steps "$POSTTRAIN_STEPS"
    "${long_context_args[@]}"
    --distributed pipeline_stage
    --nproc-per-node "$NPROC_PER_NODE"
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --pipeline-stage-schedule "$PIPELINE_STAGE_SCHEDULE"
    --pipeline-microbatches "$PIPELINE_MICROBATCHES"
    --precision fp16
    --init-dtype fp16
    --optimizer adafactor
    "${optimizer_in_backward_args[@]}"
    --optimizer-in-backward-update lowmem_adafactor
    --optimizer-in-backward-grad-clip 1.0
    --optimizer-in-backward-clip-mode rms
    --optimizer-in-backward-adafactor-chunk-rows "$OPTIMIZER_ADAFACTOR_CHUNK_ROWS"
    --optimizer-in-backward-adafactor-clip-threshold 1.0
    --optimizer-in-backward-adafactor-decay-rate -0.8
    --optimizer-in-backward-adafactor-eps1 1e-30
    "${activation_checkpointing_args[@]}"
    --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
    --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
    "${shared_posttrain_args[@]}"
    "${shared_eval_args[@]}"
    "${shared_checkpoint_readiness_args[@]}"
    "${fake_quant_args[@]}"
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
  -e OMNICODER2026_DIST_TIMEOUT_SECONDS="$DIST_TIMEOUT_SECONDS"
  -e OMNICODER2026_CHECKPOINT_SYNC_BACKEND="$CHECKPOINT_SYNC_BACKEND"
  -e OMNICODER2026_CHECKPOINT_MARKER_TIMEOUT_SECONDS="$CHECKPOINT_MARKER_TIMEOUT_SECONDS"
  -e OMNICODER2026_CHECKPOINT_MARKER_POLL_SECONDS="$CHECKPOINT_MARKER_POLL_SECONDS"
  -e OMNICODER2026_STEP_TIMING_INTERVAL="$STEP_TIMING_INTERVAL"
  -e OMNICODER2026_RANK_SKEW_INTERVAL="$RANK_SKEW_INTERVAL"
  -e OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL="$LOSS_DIAGNOSTICS_INTERVAL"
  -e OMNICODER2026_DETAILED_EVENT_LOG_INTERVAL="$DETAILED_EVENT_LOG_INTERVAL"
  -e OMNICODER2026_TIMING_CUDA_SYNC="$TIMING_CUDA_SYNC"
  -e OMNICODER2026_BLOCK_TIMING="$BLOCK_TIMING"
  -e OMNICODER2026_BLOCK_TIMING_CUDA_SYNC="$BLOCK_TIMING_CUDA_SYNC"
  -e OMNICODER2026_CHECKPOINT_DATA_HASH_POLICY="$CHECKPOINT_DATA_HASH_POLICY"
  -e OMNICODER2026_PIPELINE_REASONING_EFFORT="$PIPELINE_REASONING_EFFORT"
  -e OMNICODER2026_SKIP_FINAL_SAVE="$SKIP_FINAL_SAVE"
  -e PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
  -e OMNICODER2026_LM_LOSS_CHUNK_TOKENS="$LM_LOSS_CHUNK_TOKENS"
  -e OMNICODER2026_FFN_CHUNK_TOKENS="$FFN_CHUNK_TOKENS"
  -e OMNICODER2026_LOSS_TOKEN_STRIDE="$LOSS_TOKEN_STRIDE"
  -e OMNICODER2026_MAX_LOSS_TOKENS_PER_SAMPLE="$MAX_LOSS_TOKENS_PER_SAMPLE"
  -e OMNICODER_OPTIMIZER_IN_BACKWARD="$OPTIMIZER_IN_BACKWARD"
  -e OMNICODER_FAKE_QUANT="$FAKE_QUANT"
  -e OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF="${OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF:-0}"
  -e OMNICODER_ACTIVATION_CHECKPOINTING="$ACTIVATION_CHECKPOINTING"
  -e OMNICODER_PIPELINE_STAGE_SCHEDULE="$PIPELINE_STAGE_SCHEDULE"
  -e OMNICODER_PIPELINE_MICROBATCHES="$PIPELINE_MICROBATCHES"
  -e TOKENIZERS_PARALLELISM=false
  -e OMNICODER_ADAPTIVE_WEIGHTS="$ADAPTIVE_WEIGHTS"
  -e OMNICODER_MIXTURE_PLAN="$MIXTURE_PLAN"
  -e OMNICODER_CONTEXT_LADDER="$CONTEXT_LADDER"
  -e OMNICODER_RLVR_ALGOS="$RLVR_ALGOS"
  -e OMNICODER_EXTERNAL_CURATION_PREFLIGHT_MAX_RECORDS="${OMNICODER_EXTERNAL_CURATION_PREFLIGHT_MAX_RECORDS:-4096}"
  -e OMNICODER_DENSE_LAUNCH_PREFLIGHT_MAX_RECORDS="${OMNICODER_DENSE_LAUNCH_PREFLIGHT_MAX_RECORDS:-${OMNICODER_EXTERNAL_CURATION_PREFLIGHT_MAX_RECORDS:-4096}}"
  -e OMNICODER_POSTTRAIN_EXPLICIT_INPUTS_ONLY="${OMNICODER_POSTTRAIN_EXPLICIT_INPUTS_ONLY:-0}"
  -e OMNICODER2026_GRAD_NORM_CHUNK_ELEMS="${OMNICODER2026_GRAD_NORM_CHUNK_ELEMS:-262144}"
  -v "$REPO:/workspace"
  -v "$WEIGHTS_ROOT:/workspace/weights"
  -v /home/cereal:/home/cereal:ro
  -w /workspace
)

if [[ -d "$AI_DATA_ROOT" ]]; then
  docker_args+=(-v "$AI_DATA_ROOT:/mnt/ai_data")
fi

if [[ -n "$BENCHMARK_PREDICTION_API_KEY_ENV" && -n "${!BENCHMARK_PREDICTION_API_KEY_ENV:-}" ]]; then
  docker_args+=(-e "$BENCHMARK_PREDICTION_API_KEY_ENV")
fi

printf -v common_args_quoted "%q " "${common_args[@]}"
readiness_pre_cmd=""
if [[ -n "$CHECKPOINT_MEDIA_ROUTE_PROBE" && -z "$CHECKPOINT_READINESS_REPORT" && ! -f "$CHECKPOINT_MEDIA_ROUTE_PROBE" ]] && truthy "$AUTO_CHECKPOINT_MEDIA_ROUTE_PROBE"; then
  printf -v readiness_pre_cmd 'mkdir -p %q; python -m omnicoder.eval.media_route_probe_2026 --out %q; ' "$(dirname "$CHECKPOINT_MEDIA_ROUTE_PROBE")" "$CHECKPOINT_MEDIA_ROUTE_PROBE"
fi
run_cmd=(
  bash -lc
  "set -euo pipefail; ${readiness_pre_cmd}python -m omnicoder.training.training_orchestration_2026 ${common_args_quoted}"
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
