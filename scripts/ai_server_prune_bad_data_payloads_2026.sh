#!/usr/bin/env bash
set -euo pipefail

# Prune rejected/quarantined training payloads from AI-server dataset roots.
# Default is dry-run. Set OMNICODER_PRUNE_APPLY=1 to delete the exact manifest.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
APPLY="${OMNICODER_PRUNE_APPLY:-0}"
USE_DOCKER_ROOT="${OMNICODER_PRUNE_USE_DOCKER_ROOT:-0}"
DOCKER_IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
ACTIVE_EXTERNAL_RUN="${OMNICODER_PRUNE_ACTIVE_EXTERNAL_RUN:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${OMNICODER_PRUNE_MANIFEST:-$WEIGHTS_ROOT/data_curation_agent_2026/bad_payload_prune_manifest_${STAMP}.tsv}"

tmp="$(mktemp)"
filtered="$(mktemp)"
cleanup() {
  rm -f "$tmp" "$filtered"
}
trap cleanup EXIT

add_matches() {
  local base="$1"
  shift
  if [[ -e "$base" ]]; then
    find "$base" "$@" -type f -print 2>/dev/null >> "$tmp" || true
  fi
}

mkdir -p "$(dirname "$MANIFEST")"

# Only actual training payload rejects/quarantines are included here. Official
# benchmark caches are intentionally not scanned; some legitimate benchmark
# task names contain words like "blocked" or "rejected".
add_matches "$WEIGHTS_ROOT/data_curation_agent_2026/runs" \
  \( -path '*/rejected/*.jsonl' -o -path '*/quarantine/*.jsonl' -o -name '*.rejected.jsonl' -o -name '*rejected*.jsonl' \)
add_matches "$WEIGHTS_ROOT/data_curation_agent_2026/integrity_audits" \
  \( -name 'dataset_integrity_rejected.jsonl' -o -name 'policy_audit_rejected.jsonl' -o -name '*rejected*.jsonl' \)
add_matches "$WEIGHTS_ROOT/data_factory" \
  \( -name 'rejected_traces.jsonl' -o -name '*rejected*.jsonl' -o -path '*/quarantine/*.jsonl' \)
add_matches "$WEIGHTS_ROOT/curated_datasets_2026" \
  \( -name 'rejected_traces.jsonl' -o -name '*rejected*.jsonl' -o -path '*/quarantine/*.jsonl' \)
add_matches "$WEIGHTS_ROOT/external_datasets_2026/runs" \
  \( -name 'rejected_external.jsonl' -o -name '*rejected*.jsonl' -o -name 'blocked_until_review.jsonl' -o -name '*_blocked_until_review.jsonl' -o -path '*/quarantine/*.jsonl' \)
add_matches "$WEIGHTS_ROOT/training_orchestration_2026" \
  \( -name '*integrity_rejected*.jsonl' -o -name 'dataset_integrity_rejected.jsonl' -o -path '*/quarantine/*.jsonl' \)

if [[ -s "$tmp" ]]; then
  sort -u "$tmp" > "$filtered"
else
  : > "$filtered"
fi

if [[ -n "$ACTIVE_EXTERNAL_RUN" ]]; then
  awk -v active="$ACTIVE_EXTERNAL_RUN" 'index($0, active) != 1 {print}' "$filtered" > "$filtered.active"
  mv "$filtered.active" "$filtered"
fi

{
  printf 'size_bytes\tpath\n'
  while IFS= read -r path; do
    [[ -f "$path" ]] && printf '%s\t%s\n' "$(stat -c '%s' "$path")" "$path"
  done < "$filtered" | sort -nr
} > "$MANIFEST"

count="$(($(wc -l < "$MANIFEST") - 1))"
bytes="$(awk 'NR>1 {s+=$1} END {printf "%.0f", s+0}' "$MANIFEST")"
printf 'manifest=%s\ncount=%s\nbytes=%s\napply=%s\n' "$MANIFEST" "$count" "$bytes" "$APPLY"

if [[ "$APPLY" != "1" || "$count" -le 0 ]]; then
  exit 0
fi

if [[ "$USE_DOCKER_ROOT" == "1" ]]; then
  rel_list="${MANIFEST%.tsv}.relative-delete-list"
  awk -F '\t' -v root="$WEIGHTS_ROOT/" 'NR>1 {sub(root, "", $2); print $2}' "$MANIFEST" > "$rel_list"
  docker run --rm \
    -v "$WEIGHTS_ROOT:/weights" \
    -v "$rel_list:/delete-list:ro" \
    --entrypoint /bin/sh \
    "$DOCKER_IMAGE" \
    -c 'set -eu; while IFS= read -r rel; do case "$rel" in /*|*..*) echo "refusing suspicious path: $rel" >&2; exit 2;; *) rm -f -- "/weights/$rel";; esac; done < /delete-list'
  rm -f "$rel_list"
else
  while IFS=$'\t' read -r _size path; do
    [[ "$path" == path || -z "$path" ]] && continue
    rm -f -- "$path"
  done < "$MANIFEST"
fi

for base in \
  "$WEIGHTS_ROOT/data_curation_agent_2026/runs" \
  "$WEIGHTS_ROOT/data_curation_agent_2026/integrity_audits" \
  "$WEIGHTS_ROOT/data_factory" \
  "$WEIGHTS_ROOT/curated_datasets_2026" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs" \
  "$WEIGHTS_ROOT/training_orchestration_2026"
do
  [[ -e "$base" ]] || continue
  find "$base" -depth -type d \( -name rejected -o -name quarantine \) -empty -delete 2>/dev/null || true
done

remaining="$(
  awk -F '\t' 'NR>1 {print $2}' "$MANIFEST" | while IFS= read -r path; do
    if [[ -f "$path" ]]; then
      printf '.\n'
    fi
  done | wc -l
)"
printf 'remaining=%s\n' "$remaining"
