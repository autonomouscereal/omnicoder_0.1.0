#!/usr/bin/env bash
set -euo pipefail

ROOT="${OMNICODER_ROOT:-/workspace}"
OUT_ROOT="${OMNICODER_OUT_ROOT:-/workspace/weights/native1m_smoke}"
PRESET="${OMNICODER_PRESET:-dense_native1m_probe}"
DEVICE="${OMNICODER_DEVICE:-cuda}"
SEQ_LEN="${OMNICODER_SEQ_LEN:-128}"
STEPS="${OMNICODER_STEPS:-1}"
DATA_PATH="${OMNICODER_DATA_PATH:-/workspace/data/smoke_native1m}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" tests_logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMNICODER_STATIC_MASK_MAX="${OMNICODER_STATIC_MASK_MAX:-8192}"
export OMNICODER_PRESET="$PRESET"
export OMNICODER_STUDENT_PRESET="$PRESET"
export OMNICODER_TRAIN_PRESET="$PRESET"
export OMNICODER_TORCH_COMPILE=0
export OMNICODER_COMPILE=0
export OMNICODER_ALLOW_INDUCTOR=0
export OMNICODER_REASONER=none
export OMNI_REASONER=none
export OMNICODER_GRAPHRAG_ENABLE=0
export SFB_ENABLE=0

python3 - <<'PY'
from omnicoder.config import get_mobile_preset
p = get_mobile_preset('dense_native1m_probe')
print({
    'preset': p.name,
    'layers': p.n_layers,
    'd_model': p.d_model,
    'n_heads': p.n_heads,
    'experts': p.moe_experts,
    'max_seq_len': p.max_seq_len,
    'vocab_size': p.vocab_size,
})
assert p.max_seq_len == 1048576
assert p.moe_experts == 1
PY

python3 -m omnicoder.training.pretrain \
  --data "$DATA_PATH" \
  --seq_len "$SEQ_LEN" \
  --steps "$STEPS" \
  --batch_size 1 \
  --grad_accum 1 \
  --device "$DEVICE" \
  --mobile_preset "$PRESET" \
  --target_ctx 1048576 \
  --out "$OUT_ROOT/native1m_probe.pt" \
  --log_file "$OUT_ROOT/pretrain_log.jsonl" \
  2>&1 | tee "$OUT_ROOT/native1m_pretrain.log"

python3 - <<'PY'
from pathlib import Path
root = Path('/workspace/weights/native1m_smoke')
print({'done': True, 'artifacts': [p.name for p in root.glob('*')]})
PY
