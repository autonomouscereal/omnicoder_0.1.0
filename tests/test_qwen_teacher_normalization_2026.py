import json
import re
import sys
from pathlib import Path

from omnicoder.data_factory import curation_policy_2026 as policy


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ai_server_run_qwen_ltx_distillation_2026.sh"


def _split_qwen_text_rollouts_code() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"split_qwen_text_rollouts\(\) \{\n.*?<<'PY'\n(?P<code>.*?)\nPY\n\}",
        text,
        flags=re.DOTALL,
    )
    assert match, "split_qwen_text_rollouts embedded Python block not found"
    return match.group("code")


def _run_embedded_split(root: Path, existing: Path) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["split_qwen_text_rollouts", str(root), str(existing)]
        exec(compile(_split_qwen_text_rollouts_code(), str(SCRIPT), "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = old_argv


def test_qwen_teacher_split_sanitizes_legacy_audit_note_and_promotes_tool_payload(tmp_path: Path) -> None:
    root = tmp_path / "qwen_run"
    raw_dir = root / "raw"
    existing = tmp_path / "missing_existing"
    raw_dir.mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    teacher_signal = {
        "corrected_response": "Run pytest and report the result.",
        "corrected_tool_calls": [{"tool": "shell", "arguments": {"command": "pytest -q"}}],
        "reward": 0.83,
        "verifier_labels": [{"check": "unit_tests_pass", "passed": True}],
    }
    row = {
        "schema": "omnicoder.openai_teacher_rollout_2026.v1",
        "status": "ok",
        "record_kind": "qwen36_agentic_code_math_tool_distill",
        "input_json": {
            "messages": [
                {"role": "system", "content": "Return compact JSON."},
                {
                    "role": "user",
                    "content": "Use the shell tool. external registry row passed declared protected benchmark scan",
                },
            ],
            "source_record": {
                "modality": "tool",
                "note": "external registry row passed declared protected benchmark scan",
                "contamination": {"benchmark_name": None, "match_type": "5gram_jaccard"},
            },
        },
        "target_json": {
            "content": json.dumps(teacher_signal, sort_keys=True),
            "teacher_status": "ok",
            "teacher_signal": teacher_signal,
        },
        "modalities": ["tool", "text"],
        "quality_score": 0.95,
        "split": "train",
    }
    source = raw_dir / "qwen36_agentic_code_math_tool.raw.jsonl"
    source.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    _run_embedded_split(root, existing)

    split_rows = [
        json.loads(line)
        for line in (raw_dir / "qwen36_tool.raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(split_rows) == 1
    normalized = split_rows[0]
    assert "protected benchmark scan" not in json.dumps(normalized, sort_keys=True)
    assert "benchmark_name" not in json.dumps(normalized, sort_keys=True)
    assert normalized["target_json"]["tool_calls"] == teacher_signal["corrected_tool_calls"]
    assert normalized["target_json"]["reward"] == teacher_signal["reward"]
    assert normalized["target_json"]["verifier"] == teacher_signal["verifier_labels"]

    prompt, target = policy.message_prompt_target(normalized)
    audit = policy.audit_training_record(
        normalized,
        prompt=prompt,
        target=target,
        modality="tool",
        refs=[],
        existing_quality=normalized["quality_score"],
        config=policy.CurationPolicyConfig(min_quality_score=0.55),
    )

    assert audit["accepted"] is True
    assert "dataset_integrity:eval_leak_benchmark_marker" not in audit["reasons"]
    assert "dataset_integrity:tool_missing_valid_call_schema" not in audit["reasons"]
    assert "dataset_integrity:tool_missing_result_or_verifier" not in audit["reasons"]


def test_qwen_tool_family_is_not_downgraded_to_text_escape_hatch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_curation_family qwen36_tool tool" in text
    assert "promote_qwen_tool_teacher_payload" in text
    assert "external registry row passed declared contamination audit" in text
