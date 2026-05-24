#!/usr/bin/env bash
set -euo pipefail

cd "${OMNICODER_REPO:-/workspace/omnicoder_0.1.0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export OMNICODER_FORCE_SIMPLE_TOKENIZER="${OMNICODER_FORCE_SIMPLE_TOKENIZER:-1}"
export OMNICODER_FORBID_SIMPLE=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p data/smoke_native1m weights
if [ ! -f data/smoke_native1m/tiny.txt ]; then
  cat > data/smoke_native1m/tiny.txt <<'TXT'
Omnicoder 2026 native-1M probe: dense recurrent-linear sparse-latent trunk.
Text, tools, media codec tokens, verifier traces, and terminal actions share one ledger.
TXT
fi

python -m omnicoder.training.pretrain_2026_dense \
  --preset omnicoder2026_full_ledger_probe \
  --data data/smoke_native1m/tiny.txt \
  --out weights/omnicoder2026_full_ledger_probe.pt \
  --seq_len "${OMNICODER_2026_PROBE_SEQ_LEN:-256}" \
  --batch_size "${OMNICODER_2026_PROBE_BATCH:-1}" \
  --steps "${OMNICODER_2026_PROBE_STEPS:-2}" \
  --device "${OMNICODER_2026_PROBE_DEVICE:-cuda}"
