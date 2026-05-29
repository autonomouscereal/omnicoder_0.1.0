#!/usr/bin/env bash
set -euo pipefail

# Local-regression evaluator for a complete sharded Omnicoder 20B checkpoint.
# Outputs are engineering evidence only; they are not official/reportable scores.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
RUN_TAG_RAW="${OMNICODER_EVAL_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
RECOVERY_RUN="${OMNICODER_EVAL_RECOVERY_RUN:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_recovergrpo_d28a1d4_20260526T051052Z}"
OUT_DIR="${OMNICODER_EVAL_OUT_DIR:-weights/benchmarks_2026/post_checkpoint_eval_${RUN_TAG}}"
HOST_OUT_DIR="$WEIGHTS_ROOT/${OUT_DIR#weights/}"
LOG_DIR="$HOST_OUT_DIR/logs"

GPU_DEVICES="${OMNICODER_EVAL_GPU_DEVICES:-${OMNICODER_FAST_GPU_DEVICES:-0,4,6}}"
NPROC_PER_NODE="${OMNICODER_EVAL_NPROC_PER_NODE:-3}"
RANK_DEVICE_MAP="${OMNICODER_EVAL_RANK_DEVICE_MAP:-0,1,2}"
PLACEMENT_LAYER_COUNTS="${OMNICODER_EVAL_PLACEMENT_LAYER_COUNTS:-16,16,32}"
PRECISION="${OMNICODER_EVAL_PRECISION:-fp16}"
INIT_DTYPE="${OMNICODER_EVAL_INIT_DTYPE:-fp16}"
PRESET="${OMNICODER_EVAL_PRESET:-omnicoder2026_20b_1m}"
MAX_RECORDS_PER_FILE="${OMNICODER_EVAL_MAX_RECORDS_PER_FILE:-${OMNICODER_EVAL_MAX_RECORDS:-16}}"
HELDOUT_SEQ_LEN="${OMNICODER_EVAL_HELDOUT_SEQ_LEN:-1024}"
P40_SAFE="${OMNICODER_EVAL_P40_SAFE:-1}"
case "${P40_SAFE,,}" in
  1|true|yes|y|on) DEFAULT_CONTEXT_LADDER="${OMNICODER_EVAL_P40_CONTEXT_LADDER:-8192,32768,131072}" ;;
  *) DEFAULT_CONTEXT_LADDER="8192,32768,131072,262144,524288,1048576" ;;
esac
CONTEXT_LADDER="${OMNICODER_EVAL_CONTEXT_LADDER:-$DEFAULT_CONTEXT_LADDER}"
LONG_CONTEXT_MAX_RECORDS_PER_FILE="${OMNICODER_EVAL_LONG_CONTEXT_MAX_RECORDS_PER_FILE:-$MAX_RECORDS_PER_FILE}"
PUBLIC_DEV_TASK_ROOTS="${OMNICODER_EVAL_PUBLIC_DEV_TASK_ROOTS:-}"
BENCHMARK_PROFILE="${OMNICODER_EVAL_BENCHMARK_PROFILE:-profiles/benchmark_suite_2026.json}"
PUBLIC_DEV_MIN_TASKS="${OMNICODER_EVAL_PUBLIC_DEV_MIN_TASKS:-1}"
PREDICT_TIMEOUT_SECONDS="${OMNICODER_EVAL_PREDICT_TIMEOUT_SECONDS:-1800}"
PREDICT_MAX_OUTPUT_TOKENS="${OMNICODER_EVAL_PREDICT_MAX_OUTPUT_TOKENS:-256}"
PREDICT_MAX_PROMPT_TOKENS="${OMNICODER_EVAL_PREDICT_MAX_PROMPT_TOKENS:-4096}"
ALLOW_ONE_TOKEN_CANARY="${OMNICODER_EVAL_ALLOW_ONE_TOKEN_CANARY:-0}"
RELEASE_GATE_MIN_OUTPUT_TOKENS="${OMNICODER_EVAL_RELEASE_GATE_MIN_OUTPUT_TOKENS:-16}"
RELEASE_GATE_REQUIRED_MODALITIES="${OMNICODER_EVAL_RELEASE_GATE_MODALITIES:-}"
TARGET_DIAGNOSTICS_TOP_K="${OMNICODER_EVAL_TARGET_DIAGNOSTICS_TOP_K:-8}"
TARGET_DIAGNOSTICS_MAX_POSITIONS="${OMNICODER_EVAL_TARGET_DIAGNOSTICS_MAX_POSITIONS:-12}"
DECODE_SANITY_MODALITIES="${OMNICODER_EVAL_DECODE_SANITY_MODALITIES:-text,code,math,tool,image,video,audio,music,tts,ocr}"
DECODE_SANITY_RELEASE_GATE_MODALITIES="${OMNICODER_EVAL_DECODE_SANITY_RELEASE_GATE_MODALITIES:-text,code,tool,image,video,audio,music,tts,ocr}"
LM_LOSS_CHUNK_TOKENS="${OMNICODER_EVAL_LM_LOSS_CHUNK_TOKENS:-64}"
FAKE_QUANT_CHUNK_ROWS="${OMNICODER_EVAL_FAKE_QUANT_CHUNK_ROWS:-16}"
FAKE_QUANT_MAX_FULL_ELEMENTS="${OMNICODER_EVAL_FAKE_QUANT_MAX_FULL_ELEMENTS:-16777216}"
DIST_TIMEOUT_SECONDS="${OMNICODER_EVAL_DIST_TIMEOUT_SECONDS:-7200}"
CUDA_ALLOC_CONF="${OMNICODER_EVAL_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
PYTHON_BIN="${OMNICODER_EVAL_PYTHON:-python}"
EVIDENCE_LABEL="local-regression evidence only; not official/reportable scores"

mkdir -p "$HOST_OUT_DIR" "$LOG_DIR"

if [[ ! "$PREDICT_MAX_OUTPUT_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "OMNICODER_EVAL_PREDICT_MAX_OUTPUT_TOKENS must be an integer, got: $PREDICT_MAX_OUTPUT_TOKENS" >&2
  exit 2
fi
if (( PREDICT_MAX_OUTPUT_TOKENS <= 1 )); then
  case "${ALLOW_ONE_TOKEN_CANARY,,}" in
    1|true|yes|y|on) ;;
    *)
      echo "Refusing one-token checkpoint eval. Set OMNICODER_EVAL_ALLOW_ONE_TOKEN_CANARY=1 only for explicit non-reportable canaries." >&2
      exit 2
      ;;
  esac
fi

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

default_public_dev_task_roots() {
  local -a roots=()
  local base="$WEIGHTS_ROOT/data_factory/runs/benchmark_materialization"
  local legacy="$WEIGHTS_ROOT/official_benchmarks_2026/runs/bench_reportable_fix_eaa2463_20260525T181734Z/local_2026"
  local root
  if [[ -d "$base" ]]; then
    while IFS= read -r root; do
      [[ -n "$root" ]] || continue
      roots+=("$root")
    done < <(find "$base" -mindepth 2 -maxdepth 2 -type d -name local_2026 -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed 's/^[^ ]* //')
  fi
  if [[ -d "$legacy" ]]; then
    roots+=("$legacy")
  fi
  (IFS=','; printf '%s\n' "${roots[*]}")
}

checkpoint_is_complete_pipeline() {
  local checkpoint="$1"
  local expected_world_size="${2:-$NPROC_PER_NODE}"
  [[ -d "$checkpoint" ]] || return 1
  [[ -s "$checkpoint/manifest.json" ]] || return 1
  [[ -s "$checkpoint/.complete.json" ]] || return 1
  grep -Eq "\"world_size\"[[:space:]]*:[[:space:]]*$expected_world_size([,[:space:]}]|$)" "$checkpoint/manifest.json" || return 1
  local rank rank_file marker count
  for ((rank=0; rank<expected_world_size; rank++)); do
    printf -v rank_file 'rank%05d.pt' "$rank"
    marker="$checkpoint/${rank_file}.complete.json"
    [[ -s "$checkpoint/$rank_file" ]] || return 1
    [[ -s "$marker" ]] || return 1
  done
  count="$(find "$checkpoint" -maxdepth 1 -type f -name 'rank*.pt' | wc -l)"
  [[ "$count" -eq "$expected_world_size" ]] || return 1
}

find_latest_checkpoint() {
  local root="$RECOVERY_RUN/checkpoints/posttrain"
  local best=""
  local best_mtime=0
  local candidate mtime
  if [[ ! -d "$root" ]]; then
    return 1
  fi
  shopt -s nullglob
  for candidate in "$root"/*; do
    if checkpoint_is_complete_pipeline "$candidate" "$NPROC_PER_NODE"; then
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

infer_run_dir() {
  local path="$1"
  local cursor="$path"
  while [[ "$cursor" != "/" && -n "$cursor" ]]; do
    if [[ "$(basename "$cursor")" == "checkpoints" ]]; then
      dirname "$cursor"
      return 0
    fi
    cursor="$(dirname "$cursor")"
  done
  return 1
}

split_csv_paths() {
  local raw="$1"
  local -n out_array="$2"
  local item host
  IFS=',' read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] || continue
    host="$(host_path "$item")"
    if [[ -s "$host" || -d "$host" ]]; then
      out_array+=("$host")
    else
      echo "Missing eval input path: $host" >&2
      exit 5
    fi
  done
}

docker_eval() {
  local label="$1"
  local log="$2"
  shift 2
  local -a cmd=("$@")
  local quoted container
  printf -v quoted '%q ' "${cmd[@]}"
  container="omnicoder_post_checkpoint_eval_${RUN_TAG}_${label//[^A-Za-z0-9_.-]/_}"
  echo "container=$container" | tee "$log"
  echo "evidence_label=$EVIDENCE_LABEL" | tee -a "$log"
  docker run \
    --name "$container" \
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
    -e OMNICODER2026_LM_LOSS_CHUNK_TOKENS="$LM_LOSS_CHUNK_TOKENS" \
    -e OMNICODER2026_FAKE_QUANT_CHUNK_ROWS="$FAKE_QUANT_CHUNK_ROWS" \
    -e OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS="$FAKE_QUANT_MAX_FULL_ELEMENTS" \
    -e OMNICODER2026_FFN_CHUNK_TOKENS="${OMNICODER_EVAL_FFN_CHUNK_TOKENS:-256}" \
    -e OMNICODER_REQUIRE_HF_TOKENIZER="${OMNICODER_REQUIRE_HF_TOKENIZER:-1}" \
    -e TOKENIZERS_PARALLELISM=false \
    -v "$REPO:/workspace" \
    -v "$WEIGHTS_ROOT:/workspace/weights" \
    -v /home/cereal:/home/cereal:ro \
    -w /workspace \
    "$IMAGE" \
    bash -lc "set -euo pipefail; $quoted" 2>&1 | tee -a "$log"
}

CHECKPOINT_INPUT="${OMNICODER_EVAL_CHECKPOINT:-}"
if [[ -z "$PUBLIC_DEV_TASK_ROOTS" ]]; then
  PUBLIC_DEV_TASK_ROOTS="$(default_public_dev_task_roots)"
fi
if [[ -z "$CHECKPOINT_INPUT" ]]; then
  CHECKPOINT_INPUT="$(find_latest_checkpoint)" || {
    echo "No complete ${NPROC_PER_NODE}-shard checkpoint found under $RECOVERY_RUN/checkpoints/posttrain." >&2
    exit 4
  }
fi
CHECKPOINT_HOST="$(host_path "$CHECKPOINT_INPUT")"
CHECKPOINT_CONTAINER="$(container_path "$CHECKPOINT_HOST")"

if ! checkpoint_is_complete_pipeline "$CHECKPOINT_HOST" "$NPROC_PER_NODE"; then
  echo "Checkpoint is not a complete ${NPROC_PER_NODE}-shard Omnicoder pipeline checkpoint: $CHECKPOINT_HOST" >&2
  exit 4
fi

RUN_DIR="$(infer_run_dir "$CHECKPOINT_HOST" || true)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  RUN_DIR="$RECOVERY_RUN"
fi

cat > "$HOST_OUT_DIR/README.local_regression.txt" <<EOF
$EVIDENCE_LABEL

This directory was produced by scripts/ai_server_run_post_checkpoint_eval_20b.sh.
It validates a complete 3-shard Omnicoder 20B checkpoint, runs heldout pipeline
sample loss, runs target-token rank/CE diagnostics, runs decode sanity probes
for text/code/math/tool/image/video/audio/music/tts/ocr, runs context-budget
plus long-context sample-loss probes over the configured ladder, and runs local
public-dev checkpoint predictions only when task JSONL files are present. These
artifacts are for regression evidence and engineering triage only.
EOF

cat > "$HOST_OUT_DIR/checkpoint_validation.local_regression.json" <<EOF
{
  "schema": "omnicoder.post_checkpoint_eval_20b.checkpoint_validation.v1",
  "evidence_label": "$EVIDENCE_LABEL",
  "official_score": false,
  "checkpoint_host": "$CHECKPOINT_HOST",
  "checkpoint_container": "$CHECKPOINT_CONTAINER",
  "preset": "$PRESET",
  "expected_world_size": 3,
  "status": "passed"
}
EOF

HELDOUT_HOST_FILES=()
if [[ -n "${OMNICODER_EVAL_HELDOUT_DATA:-}" ]]; then
  split_csv_paths "$OMNICODER_EVAL_HELDOUT_DATA" HELDOUT_HOST_FILES
else
  for candidate in \
    "$RUN_DIR/jsonl/eval_all_modalities.jsonl" \
    "$RUN_DIR/jsonl/test_all_modalities.jsonl"; do
    [[ -s "$candidate" ]] && HELDOUT_HOST_FILES+=("$candidate")
  done
fi

if [[ "${#HELDOUT_HOST_FILES[@]}" -eq 0 ]]; then
  echo "No heldout eval/test JSONL files found. Set OMNICODER_EVAL_HELDOUT_DATA." >&2
  exit 6
fi

LONG_CONTEXT_HOST_FILES=()
if [[ -n "${OMNICODER_EVAL_LONG_CONTEXT_DATA:-}" ]]; then
  split_csv_paths "$OMNICODER_EVAL_LONG_CONTEXT_DATA" LONG_CONTEXT_HOST_FILES
else
  for candidate in \
    "$RUN_DIR/jsonl/eval_long_context.jsonl" \
    "$RUN_DIR/jsonl/test_long_context.jsonl"; do
    [[ -s "$candidate" ]] && LONG_CONTEXT_HOST_FILES+=("$candidate")
  done
fi

if [[ "${#LONG_CONTEXT_HOST_FILES[@]}" -eq 0 ]]; then
  echo "No long-context eval/test JSONL files found. Set OMNICODER_EVAL_LONG_CONTEXT_DATA." >&2
  exit 7
fi

{
  echo "$EVIDENCE_LABEL"
  printf 'checkpoint=%s\n' "$CHECKPOINT_HOST"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'heldout_files:\n'
  printf '  %s\n' "${HELDOUT_HOST_FILES[@]}"
  printf 'long_context_files:\n'
  printf '  %s\n' "${LONG_CONTEXT_HOST_FILES[@]}"
} > "$HOST_OUT_DIR/input_files.local_regression.txt"

heldout_args=()
for path in "${HELDOUT_HOST_FILES[@]}"; do
  heldout_args+=(--data "$(container_path "$path")")
done
long_context_args=()
for path in "${LONG_CONTEXT_HOST_FILES[@]}"; do
  long_context_args+=(--data "$(container_path "$path")")
done

sample_loss_common=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node "$NPROC_PER_NODE"
  --max_restarts 0
  -m omnicoder.eval.pipeline_sample_loss_2026
  --checkpoint "$CHECKPOINT_CONTAINER"
  --preset "$PRESET"
  --rank_device_map "$RANK_DEVICE_MAP"
  --placement_layer_counts "$PLACEMENT_LAYER_COUNTS"
  --precision "$PRECISION"
  --init-dtype "$INIT_DTYPE"
  --fake_quant
  --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
  --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
  --lm-loss-chunk-tokens "$LM_LOSS_CHUNK_TOKENS"
  --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS"
  --require_target_contract
  --allow-p40-target-contract-eval
)

target_diagnostics_common=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node "$NPROC_PER_NODE"
  --max_restarts 0
  -m omnicoder.eval.pipeline_target_token_diagnostics_2026
  --checkpoint "$CHECKPOINT_CONTAINER"
  --preset "$PRESET"
  --rank-device-map "$RANK_DEVICE_MAP"
  --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
  --precision "$PRECISION"
  --init-dtype "$INIT_DTYPE"
  --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS"
  --expected-world-size "$NPROC_PER_NODE"
  --fake-quant
  --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS"
  --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS"
  --require-target-contract
  --allow-p40-target-contract-eval
)

docker_eval heldout_sample_loss "$LOG_DIR/heldout_sample_loss.log" \
  "${sample_loss_common[@]}" \
  --seq-len "$HELDOUT_SEQ_LEN" \
  --max-records-per-file "$MAX_RECORDS_PER_FILE" \
  --out "/workspace/$OUT_DIR/heldout_pipeline_sample_loss.local_regression.json" \
  "${heldout_args[@]}"

docker_eval heldout_target_token_diagnostics "$LOG_DIR/heldout_target_token_diagnostics.log" \
  "${target_diagnostics_common[@]}" \
  --seq-len "$HELDOUT_SEQ_LEN" \
  --max-records-per-file "$MAX_RECORDS_PER_FILE" \
  --top-k "$TARGET_DIAGNOSTICS_TOP_K" \
  --max-positions "$TARGET_DIAGNOSTICS_MAX_POSITIONS" \
  --out "/workspace/$OUT_DIR/heldout_target_token_diagnostics.local_regression.json" \
  "${heldout_args[@]}"

IFS=',' read -r -a context_rungs <<< "$CONTEXT_LADDER"
for context_len in "${context_rungs[@]}"; do
  context_len="${context_len#"${context_len%%[![:space:]]*}"}"
  context_len="${context_len%"${context_len##*[![:space:]]}"}"
  [[ -n "$context_len" ]] || continue
  docker_eval "context_budget_${context_len}" "$LOG_DIR/context_budget_${context_len}.log" \
    "$PYTHON_BIN" -m omnicoder.inference.context_budget_2026 \
    --profile "$PRESET" \
    --context "$context_len" \
    --out "/workspace/$OUT_DIR/long_context_budget_ctx${context_len}.local_regression.json"
  docker_eval "long_context_probe_${context_len}" "$LOG_DIR/long_context_probe_${context_len}.log" \
    "${sample_loss_common[@]}" \
    --seq-len "$context_len" \
    --max-records-per-file "$LONG_CONTEXT_MAX_RECORDS_PER_FILE" \
    --out "/workspace/$OUT_DIR/long_context_probe_ctx${context_len}.local_regression.json" \
    "${long_context_args[@]}"
done

DECODE_SANITY_TASKS_HOST="$HOST_OUT_DIR/decode_sanity_tasks.local_regression.jsonl"
cat > "$DECODE_SANITY_TASKS_HOST" <<'EOF'
{"benchmark_id":"decode_sanity_text_2026","task_id":"text-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"text","output_field":"prediction","prompt":"Write one complete sentence explaining why checkpoint decode sanity is checked."}
{"benchmark_id":"decode_sanity_code_2026","task_id":"code-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"code","output_field":"prediction","prompt":"Return a short Python function named add_two_numbers with one return statement."}
{"benchmark_id":"decode_sanity_math_2026","task_id":"math-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"math","output_field":"prediction","prompt":"Answer with a short calculation: what is 17 plus 25?"}
{"benchmark_id":"decode_sanity_tool_2026","task_id":"tool-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"tool","output_field":"tool_call","task_format":"tool_call_json","prompt":"Return one compact JSON tool call for a search tool with query checkpoint decode sanity."}
{"benchmark_id":"decode_sanity_image_generation_2026","task_id":"image-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"image","output_modality":"image","output_field":"generated_artifact","prompt":"Generate diagnostic image artifact tokens for a small blue square on a white background."}
{"benchmark_id":"decode_sanity_video_generation_2026","task_id":"video-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"video","output_modality":"video","output_field":"generated_artifact","prompt":"Generate diagnostic video artifact tokens for a one second clip of a dot moving left to right."}
{"benchmark_id":"decode_sanity_audio_generation_2026","task_id":"audio-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"audio","output_modality":"audio","output_field":"generated_artifact","prompt":"Generate diagnostic audio artifact tokens for a brief clear tone."}
{"benchmark_id":"decode_sanity_music_generation_2026","task_id":"music-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"music","output_modality":"music","output_field":"generated_artifact","prompt":"Generate diagnostic music artifact tokens for a two bar upbeat piano loop."}
{"benchmark_id":"decode_sanity_tts_generation_2026","task_id":"tts-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"tts","output_modality":"tts","output_field":"generated_artifact","prompt":"Generate diagnostic TTS speech artifact tokens saying checkpoint decode sanity passed."}
{"benchmark_id":"decode_sanity_text_extraction_2026","task_id":"text-extraction-1","reportable":false,"dataset_revision":"local-regression-2026-decode-sanity","snapshot_id":"local-regression-2026-decode-sanity","source":"omnicoder_local_decode_sanity_2026","modality":"ocr","output_modality":"text","output_field":"prediction","prompt":"OCR probe: read the imagined image text CHECKPOINT OK and return only the extracted text."}
EOF

{
  printf 'decode_sanity_modalities=%s\n' "$DECODE_SANITY_MODALITIES"
  printf 'decode_sanity_tasks=%s\n' "$DECODE_SANITY_TASKS_HOST"
} >> "$HOST_OUT_DIR/input_files.local_regression.txt"

docker_eval decode_sanity_predictions "$LOG_DIR/decode_sanity_predictions.log" \
  "$PYTHON_BIN" -m omnicoder.eval.pipeline_checkpoint_batch_predict_2026 \
  --checkpoint "$CHECKPOINT_CONTAINER" \
  --tasks "/workspace/$OUT_DIR/decode_sanity_tasks.local_regression.jsonl" \
  --out "/workspace/$OUT_DIR/decode_sanity_predictions.local_regression.jsonl" \
  --summary "/workspace/$OUT_DIR/decode_sanity_prediction_summary.local_regression.json" \
  --model "$CHECKPOINT_CONTAINER" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --rank-device-map "$RANK_DEVICE_MAP" \
  --placement-layer-counts "$PLACEMENT_LAYER_COUNTS" \
  --precision "$PRECISION" \
  --init-dtype "$INIT_DTYPE" \
  --max-prompt-tokens "$PREDICT_MAX_PROMPT_TOKENS" \
  --max-output-tokens "$PREDICT_MAX_OUTPUT_TOKENS" \
  --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS" \
  --fake-quant \
  --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS" \
  --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS" \
  --require-target-contract \
  --allow-p40-target-contract-eval \
  --allow-local-dev-tasks \
  --force

docker_eval decode_sanity_release_gate "$LOG_DIR/decode_sanity_release_gate.log" \
  "$PYTHON_BIN" -m omnicoder.eval.omnimodal_release_gate_2026 \
  --predictions "/workspace/$OUT_DIR/decode_sanity_predictions.local_regression.jsonl" \
  --out "/workspace/$OUT_DIR/decode_sanity_release_gate.local_regression.json" \
  --min-output-tokens "$RELEASE_GATE_MIN_OUTPUT_TOKENS" \
  --require-modalities "$DECODE_SANITY_RELEASE_GATE_MODALITIES"

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

if [[ "${#public_dev_roots_present[@]}" -gt 0 ]]; then
  public_task_args=()
  score_task_args=()
  for root in "${public_dev_roots_present[@]}"; do
    public_task_args+=(--tasks "$(container_path "$root")")
    score_task_args+=(--tasks "$(container_path "$root")")
  done
  docker_eval public_dev_predictions "$LOG_DIR/public_dev_predictions.log" \
    "$PYTHON_BIN" -m omnicoder.eval.pipeline_checkpoint_batch_predict_2026 \
    --checkpoint "$CHECKPOINT_CONTAINER" \
    "${public_task_args[@]}" \
    --out "/workspace/$OUT_DIR/public_dev_predictions.local_regression.jsonl" \
    --summary "/workspace/$OUT_DIR/public_dev_prediction_summary.local_regression.json" \
    --model "$CHECKPOINT_CONTAINER" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --rank-device-map "$RANK_DEVICE_MAP" \
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS" \
    --precision "$PRECISION" \
    --init-dtype "$INIT_DTYPE" \
    --max-prompt-tokens "$PREDICT_MAX_PROMPT_TOKENS" \
    --max-output-tokens "$PREDICT_MAX_OUTPUT_TOKENS" \
    --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS" \
    --fake-quant \
    --fake-quant-chunk-rows "$FAKE_QUANT_CHUNK_ROWS" \
    --fake-quant-max-full-elements "$FAKE_QUANT_MAX_FULL_ELEMENTS" \
    --require-target-contract \
    --allow-p40-target-contract-eval \
    --allow-local-dev-tasks \
    --force
  release_gate_args=(
    --predictions "/workspace/$OUT_DIR/public_dev_predictions.local_regression.jsonl"
    --out "/workspace/$OUT_DIR/public_dev_release_gate.local_regression.json"
    --min-output-tokens "$RELEASE_GATE_MIN_OUTPUT_TOKENS"
  )
  if [[ -n "$RELEASE_GATE_REQUIRED_MODALITIES" ]]; then
    release_gate_args+=(--require-modalities "$RELEASE_GATE_REQUIRED_MODALITIES")
  else
    release_gate_args+=(--require-modalities "")
  fi
  docker_eval public_dev_release_gate "$LOG_DIR/public_dev_release_gate.log" \
    "$PYTHON_BIN" -m omnicoder.eval.omnimodal_release_gate_2026 \
    "${release_gate_args[@]}"
  docker_eval public_dev_scoring "$LOG_DIR/public_dev_scoring.log" \
    "$PYTHON_BIN" -m omnicoder.eval.benchmark_suite_2026 \
    --profile "$BENCHMARK_PROFILE" \
    --out-dir "/workspace/$OUT_DIR/public_dev_scoring.local_regression" \
    --model "$CHECKPOINT_CONTAINER" \
    run-reportable \
    "${score_task_args[@]}" \
    --predictions "/workspace/$OUT_DIR/public_dev_predictions.local_regression.jsonl" \
    --cycle smoke \
    --run-id "local_regression_${RUN_TAG}" \
    --min-tasks "$PUBLIC_DEV_MIN_TASKS" \
    --missing-reportable-policy allow
else
  cat > "$HOST_OUT_DIR/public_dev_prediction_summary.local_regression.json" <<EOF
{
  "schema": "omnicoder.post_checkpoint_eval_20b.public_dev_predictions.v1",
  "evidence_label": "$EVIDENCE_LABEL",
  "official_score": false,
  "status": "skipped",
  "reason": "no_local_public_dev_task_jsonl",
  "checked_roots": "$PUBLIC_DEV_TASK_ROOTS"
}
EOF
fi

cat > "$HOST_OUT_DIR/local_regression_manifest.json" <<EOF
{
  "schema": "omnicoder.post_checkpoint_eval_20b.local_regression_manifest.v1",
  "evidence_label": "$EVIDENCE_LABEL",
  "official_score": false,
  "run_tag": "$RUN_TAG",
  "repo": "$REPO",
  "weights_root": "$WEIGHTS_ROOT",
  "checkpoint": "$CHECKPOINT_HOST",
  "out_dir": "$HOST_OUT_DIR",
  "gpu_devices": "$GPU_DEVICES",
  "p40_safe": "$P40_SAFE",
  "dist_timeout_seconds": "$DIST_TIMEOUT_SECONDS",
  "heldout_max_records_per_file": "$MAX_RECORDS_PER_FILE",
  "long_context_max_records_per_file": "$LONG_CONTEXT_MAX_RECORDS_PER_FILE",
  "context_ladder": "$CONTEXT_LADDER",
  "target_diagnostics": "/workspace/$OUT_DIR/heldout_target_token_diagnostics.local_regression.json",
  "decode_sanity_modalities": "$DECODE_SANITY_MODALITIES",
  "decode_sanity_tasks": "/workspace/$OUT_DIR/decode_sanity_tasks.local_regression.jsonl",
  "decode_sanity_predictions": "/workspace/$OUT_DIR/decode_sanity_predictions.local_regression.jsonl",
  "decode_sanity_release_gate": "/workspace/$OUT_DIR/decode_sanity_release_gate.local_regression.json",
  "status": "completed"
}
EOF

echo "post_checkpoint_eval_out=$HOST_OUT_DIR"
echo "evidence_label=$EVIDENCE_LABEL"
