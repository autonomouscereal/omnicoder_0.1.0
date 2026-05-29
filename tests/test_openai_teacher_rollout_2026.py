from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import openai_teacher_rollout_2026 as rollout


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_teacher_rollout_resume_skips_existing_rows(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out = tmp_path / "teacher.jsonl"
    _jsonl(
        source,
        [
            {"input_json": {"messages": [{"role": "user", "content": "first"}]}},
            {"input_json": {"messages": [{"role": "user", "content": "second"}]}},
            {"input_json": {"messages": [{"role": "user", "content": "third"}]}},
        ],
    )
    _jsonl(
        out,
        [
            {
                "schema": "omnicoder.openai_teacher_rollout_2026.v1",
                "index": 1,
                "status": "ok",
                "target_json": {"content": "already done"},
            }
        ],
    )

    prompts: list[str] = []

    def fake_post_chat(base_url: str, model: str, prompt: str, timeout: int, max_tokens: int, temperature: float) -> dict:
        prompts.append(prompt)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "corrected_response": (
                                    "This is a substantive teacher correction with useful detail for the requested prompt: "
                                    + prompt
                                )
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(rollout, "post_chat", fake_post_chat)
    code = rollout.main(
        [
            "--input",
            str(source),
            "--out",
            str(out),
            "--limit",
            "3",
            "--resume",
        ]
    )

    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["index"] for row in rows] == [1, 2, 3]
    assert len(prompts) == 2
    assert "second" in prompts[0]
    assert "third" in prompts[1]


def test_teacher_rollout_rejects_non_json_or_refusal_outputs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out = tmp_path / "teacher.jsonl"
    _jsonl(
        source,
        [
            {"input_json": {"messages": [{"role": "user", "content": "write a careful solution"}]}},
            {"input_json": {"messages": [{"role": "user", "content": "write another careful solution"}]}},
        ],
    )

    responses = [
        "plain prose without the required JSON object",
        json.dumps({"corrected_response": "As an AI model, I cannot assist with that request."}),
    ]

    def fake_post_chat(base_url: str, model: str, prompt: str, timeout: int, max_tokens: int, temperature: float) -> dict:
        return {"choices": [{"message": {"content": responses.pop(0)}}]}

    monkeypatch.setattr(rollout, "post_chat", fake_post_chat)
    code = rollout.main(["--input", str(source), "--out", str(out), "--limit", "2"])

    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["status"] for row in rows] == ["failed", "failed"]
    assert [row["quality_score"] for row in rows] == [0.0, 0.0]
    assert rows[0]["error"] == "teacher_signal_not_json"
    assert rows[1]["error"].startswith("teacher_signal_bad_marker:")


def test_teacher_rollout_defaults_to_exact_lmstudio_qwen_lane(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out = tmp_path / "teacher.jsonl"
    _jsonl(source, [{"input_json": {"messages": [{"role": "user", "content": "write a useful solution"}]}}])
    calls: list[tuple[str, str]] = []

    def fake_post_chat(base_url: str, model: str, prompt: str, timeout: int, max_tokens: int, temperature: float) -> dict:
        calls.append((base_url, model))
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "corrected_response": (
                                    "This is a substantive exact-lane teacher response with enough useful detail."
                                )
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(rollout, "post_chat", fake_post_chat)
    code = rollout.main(["--input", str(source), "--out", str(out), "--limit", "1"])

    assert code == 0
    assert calls == [("http://127.0.0.1:1234/v1", "qwen/qwen3.6-27b")]
