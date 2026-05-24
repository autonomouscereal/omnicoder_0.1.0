# Omnicoder 2026 Agentic Tool-Calling Training

This document defines the agentic tool-calling training lane for Omnicoder 2026.
The implementation entry point is:

```powershell
agentic-tool-train-2026 --help
```

It is also wired into `full-harness-2026` as the `agentic_tool_training`
stage, which runs after `export_sft` and before teacher jobs / post-training
handoff.

The lane trains the model to plan, call tools, observe results, recover from
failures, and stop safely. It uses 2025-2026 trace formats and the repo's
JSONL-first harness. Raw PostgreSQL mirroring is allowed through schema files,
while the training path stays JSONL-first.

## Source Traces

Accepted sources are immutable tool traces with enough provenance to reproduce
or audit the action:

- Codex, Hermes, Claude Code, and dashboard harness sessions from
  `agent_memory_events`.
- Tool-call audit rows from the PostgreSQL agent memory backend.
- Full-harness data-factory exports under `weights/data_factory/...`.
- AI server service traces for approved reference modules such as ComfyUI,
  GitLab, Keycloak, Mailcow, Wazuh, iTop, SearXNG, and the AI proxy.
- Synthetic teacher traces generated from accepted real traces for repair,
  masking, negative sampling, and reward labeling.

Every source row must preserve `source`, `source_date`, `trace_id`,
`session_id`, `turn_index`, `event_type`, `created_at`, `tool_name`,
`tool_args`, `tool_result`, `status`, and `provenance`. Secret-bearing rows are
redacted and rejected from training exports by default.

## JSONL Formats

### Raw Event JSONL

Raw rows are append-only. They may include provider-specific fields, but the
normalizer must be able to derive the common fields.

```json
{"schema":"omnicoder.agent_tool.raw.v1","source":"codex","source_date":"2026-05-23","trace_id":"tr_001","session_id":"sess_001","turn_index":7,"event_type":"tool_call","tool_name":"shell_command","tool_args":{"command":"pytest tests/test_tool_use.py"},"tool_result":{"exit_code":0,"stdout":"..."},"status":"ok","provenance":{"host":"workstation","collector":"memory_trace_collectors_2026"}}
```

### Normalized Tool-Turn JSONL

Normalized rows are the curation contract consumed by quality scoring,
contamination scanning, SFT export, and teacher-job creation.

```json
{"schema":"omnicoder.agent_tool.normalized.v1","id":"toolturn_tr_001_0007","trace_id":"tr_001","session_id":"sess_001","messages":[{"role":"user","content":"Run the tests."},{"role":"assistant","content":"I will run the focused test target first.","tool_calls":[{"id":"call_001","name":"shell_command","arguments":{"command":"pytest tests/test_tool_use.py"}}]},{"role":"tool","tool_call_id":"call_001","name":"shell_command","content":{"exit_code":0,"stdout":"..."}}],"labels":{"accepted":true,"quality":0.82,"safety":"clean","contamination":"clear","split":"train"},"provenance":{"raw_id":"...","hash":"..."}}
```

### SFT Chat JSONL

SFT exports use chat-style rows with explicit tool-call messages. This trains
format, argument construction, observation reading, and final response behavior.

```json
{"messages":[{"role":"system","content":"Use available tools only when needed. Do not expose secrets."},{"role":"user","content":"Check the service health."},{"role":"assistant","content":"","tool_calls":[{"id":"call_001","type":"function","function":{"name":"server_manager_execute","arguments":"{\"server\":\"ai\",\"command\":\"docker compose ps\"}"}}]},{"role":"tool","tool_call_id":"call_001","content":"{\"status\":\"ok\",\"services\":[...]}"} ,{"role":"assistant","content":"The services are running and the health check passed."}],"metadata":{"trace_id":"tr_001","curriculum_stage":"tool_sft_foundation","source_date":"2026-05-23"}}
```

### Preference And Reward JSONL

Reward rows compare safe, successful tool use against unsafe, wasteful, or
incorrect behavior.

```json
{"schema":"omnicoder.agent_tool.preference.v1","prompt":"Restart only the dashboard UI service.","chosen":{"tool_calls":[{"name":"server_manager_execute","arguments":{"server":"ai","command":"docker compose restart dashboard-ui"}}],"final":"Dashboard UI restarted."},"rejected":{"tool_calls":[{"name":"server_manager_execute","arguments":{"server":"ai","command":"docker compose down"}}],"final":"All services restarted."},"reward_axes":{"task_success":1.0,"least_privilege":1.0,"blast_radius":1.0,"secret_safety":1.0},"source_date":"2026-05-23"}
```

## Curriculum Stages

1. `trace_ingest`: collect raw event JSONL from memory, harness logs, AI server
   service logs, and approved synthetic generators.
2. `normalize`: convert provider-specific event rows into normalized
   tool-turn rows with stable ids and provenance hashes.
3. `quality_score`: reject empty, duplicated, malformed, secret-bearing,
   non-reproducible, or low-signal rows.
4. `contam_scan`: remove protected benchmark prompts, rubrics, private eval
   artifacts, and tool transcripts that reveal expected eval answers.
5. `tool_sft_foundation`: train message format, tool-call JSON, observation
   reading, stop conditions, and final-response style.
6. `schema_masked_sft`: hide selected tool names, parameter names, optional
   fields, or result details so the model learns intent and schema inference
   instead of memorizing exact wrappers.
7. `repair_and_recovery`: teach invalid JSON repair, missing-argument recovery,
   tool-not-found fallback, timeout handling, retry budgets, and graceful
   escalation.
8. `safety_negatives`: contrast safe least-privilege actions with destructive,
   secret-exposing, credential-leaking, policy-bypassing, or over-broad actions.
9. `teacher_critique`: use local teachers to generate critiques, corrected
   tool calls, preference pairs, verifier labels, and process-reward labels.
10. `reward_modeling`: train outcome and process rewards for task success,
    schema validity, safety, trace efficiency, and observation grounding.
11. `rl_agentic`: run GRPO/DAPO/Tree-GRPO-style rollouts against verifiable
    local tasks, shell sandboxes, tests, service health checks, and tool-state
    validators.
12. `q4_recovery`: distill and recover behavior under q4 deployment targets,
    long-context traces, and mobile/edge context budgets.

## Tool Schema Masking

Schema masking creates variants of accepted traces without changing the
underlying task. It prevents brittle memorization and prepares the model for new
tools.

- Mask tool names: `server_manager_execute` becomes `tool_a` while metadata
  keeps the real name outside the loss target.
- Mask optional fields: hide fields such as `timeout_ms` or `workdir` when the
  task does not require them.
- Mask argument names: replace `command` with `arg_1` for schema-inference
  examples, then reveal the real schema in later turns.
- Mask result detail: keep exit code and summary while dropping long stdout.
- Mask unavailable tools: present a narrower tool list and require fallback to
  available tools.

Training exports must store both `visible_schema` and `full_schema_hash` so the
model trains on the masked view while audit tooling can prove which real schema
was used.

## Safety Negatives

Safety negatives are explicit rejected examples. They should be close to the
positive behavior so the reward model learns the boundary.

- Destructive action when a narrow read-only check was requested.
- Raw SSH, hardcoded password, plaintext token, or secret echoed into logs.
- Hidden privilege escalation or bypass of approval-gated operations.
- Over-broad service restart when a single process or container is enough.
- Tool call with hallucinated tool name, malformed JSON, or invented argument.
- Ignoring a failed tool result and claiming success.
- Training on protected eval artifacts or benchmark answer leakage.
- Retrying the same failing command without changing evidence or hypothesis.

Safety labels should include `secret_safety`, `least_privilege`,
`blast_radius`, `schema_validity`, `observation_grounding`,
`contamination_clear`, and `requires_approval`.

## RL And Reward Phases

The reward stack should be staged before on-policy RL:

1. Outcome reward: task completed, tests pass, service state matches request,
   or artifact exists.
2. Process reward: correct tool choice, valid JSON, minimal commands, evidence
   gathering before edits, and concise user updates.
3. Safety reward: no secrets, no raw SSH when server-manager is required, no
   destructive broad operations, and no protected-data leakage.
4. Tool-state reward: validator checks the actual filesystem, test output,
   API result, service status, or JSONL registry event.
5. Preference optimization: DPO/ORPO/KTO/SimPO-style training over chosen and
   rejected tool traces.
6. Verifiable RL: GRPO/DAPO/Tree-GRPO rollouts with deterministic task checks,
   retry limits, and automatic failure labels.

## Domain-Specific RLVR Exports

The 2026 lane now emits separate verifier-ready RLVR files in addition to the
legacy aggregate `tool_rlvr.jsonl`:

- `math_rlvr.jsonl`: final-answer exactness, symbolic equivalence, numeric
  tolerance, and benchmark-answer-key exclusion.
- `code_rlvr.jsonl`: unit-test pass rate, patch application, dependency
  safety, minimal patching, and runtime behavior.
- `terminal_rlvr.jsonl`: exit code, stdout/stderr match, filesystem state,
  recovery after errors, and destructive-action boundaries.
- `browser_rlvr.jsonl`: answer exactness, citation support, page state,
  navigation efficiency, and source freshness.
- `tool_rlvr.jsonl`: tool schema validity, argument exactness, state update
  consistency, and task outcome.

`profiles/agentic_tool_training_2026.json` carries `rlvr_domains` and
`reward_weights`, so math, coding, terminal, browser, and generic tool rows can
be sampled and optimized independently. This is the route for agentic math and
coding reinforcement learning without smearing every task into one generic
tool-use reward.

Use `posttrain-bridge-2026` for dry-run validation and algorithm handoff. Use
teacher jobs for critique and reward-label generation before expensive RL.

## AI Server Commands

Use the server-manager v2 client for AI server operations. Do not use raw SSH or
hardcoded credentials.

```powershell
python "C:/Users/cereal/.Codex/skills/server-manager/ssh_client.py" --server ai --execute "hostname"
```

Dry-run the harness on the AI server from the repo root:

```powershell
python "C:/Users/cereal/.Codex/skills/server-manager/ssh_client.py" --server ai --script '
cd /path/to/omnicoder_0.1.0
python -m omnicoder.training.full_harness_2026 run \
  --profile profiles/training_harness_2026.json \
  --trace-input data/raw/agent_memory_events_2026.jsonl \
  --stages ingest_trace,quality_score,contam_scan,export_sft,agentic_tool_training,teacher_jobs \
  --dry-run
'
```

Build only the agentic tool training artifacts:

```powershell
python "C:/Users/cereal/.Codex/skills/server-manager/ssh_client.py" --server ai --script '
cd /path/to/omnicoder_0.1.0
python -m omnicoder.training.agentic_tool_training_2026 \
  --profile profiles/agentic_tool_training_2026.json \
  build \
  --input weights/data_factory/trace_orchestrator_2026/jsonl/contamination_scanned.jsonl \
  --out-dir weights/agentic_tool_training_2026 \
  --dry-run
'
```

Run the local teacher-job curriculum after curation:

```powershell
python "C:/Users/cereal/.Codex/skills/server-manager/ssh_client.py" --server ai --script '
cd /path/to/omnicoder_0.1.0
python -m omnicoder.training.distillation_curriculum_2026 all \
  --profile profiles/distillation_curriculum_2026.json \
  --records weights/data_factory/trace_orchestrator_2026/jsonl/contamination_scanned.jsonl \
  --out-dir weights/distillation_2026
'
```

Validate post-training handoff without launching a long run:

```powershell
python "C:/Users/cereal/.Codex/skills/server-manager/ssh_client.py" --server ai --script '
cd /path/to/omnicoder_0.1.0
python -m omnicoder.training.posttrain_bridge_2026 \
  --algorithm grpo \
  --train_jsonl weights/data_factory/trace_orchestrator_2026/exports/sft_traces.jsonl \
  --out_dir weights/posttrain_2026/grpo \
  --dry_run --check_deps
'
```

## Full-Harness-2026 Integration

`full-harness-2026` is the canonical orchestration entry point. Agentic
tool-calling data enters at `ingest_trace`, passes through `quality_score` and
`contam_scan`, exports grouped conversations at `export_sft`, builds
tool-specific SFT/preference/reward/RLVR artifacts at `agentic_tool_training`,
optionally builds `teacher_jobs`, then feeds `sft_qlora_bridge`,
`native_train`, `eval_smoke`, and `context_budget`.

Recommended workstation smoke:

```powershell
python -m omnicoder.training.full_harness_2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages ingest_trace,quality_score,contam_scan,export_sft,agentic_tool_training,teacher_jobs `
  --dry-run
```

Installed console-script equivalent:

```powershell
full-harness-2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages ingest_trace,quality_score,contam_scan,export_sft,agentic_tool_training,teacher_jobs `
  --dry-run
```

Inspect the run registry:

```powershell
python -m omnicoder.training.run_registry_2026 status --run-id <run_id>
```

## Validation Checklist

- The doc exists at `docs/AgenticToolTraining2026.md`.
- `pyproject.toml` exposes `agentic-tool-train-2026`.
- JSON examples parse as single-line JSON objects.
- The doc references only existing 2026 entry points.
- No secrets or direct credential handling workflows are introduced.
