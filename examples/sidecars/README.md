# SFB Sidecar Examples

These JSONL files demonstrate optional sidecar inputs for Semantic‑Factoring Brain (SFB):

- `clip_pairs.jsonl` – image/text pairs for CLIPScore. Set `SFB_CLIP_JSONL` to this file.
- `asr_pairs.jsonl` – ASR references (and optionally hyps). Set `SFB_ASR_JSONL`.
- `code_tasks.jsonl` – PAL-style code eval inputs. Set `SFB_CODE_TASKS_JSONL`.

Example usage (bash):

```
export SFB_ENABLE=1
export SFB_CLIP_JSONL=examples/sidecars/clip_pairs.jsonl
export SFB_ASR_JSONL=examples/sidecars/asr_pairs.jsonl
export SFB_CODE_TASKS_JSONL=examples/sidecars/code_tasks.jsonl
```

Then run your normal generator or tests. Missing deps degrade to no‑ops.
