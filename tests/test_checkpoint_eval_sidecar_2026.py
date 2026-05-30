from __future__ import annotations

import argparse
import json
import subprocess

from omnicoder.eval import checkpoint_eval_sidecar_2026 as sidecar


def _args(tmp_path) -> argparse.Namespace:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "rank00000.pt").write_bytes(b"rank0")
    (checkpoint / "manifest.json").write_text('{"status":"complete","world_size":1}\n', encoding="utf-8")
    (checkpoint / ".complete.json").write_text('{"status":"complete","world_size":1}\n', encoding="utf-8")
    data = tmp_path / "heldout.jsonl"
    data.write_text('{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n', encoding="utf-8")
    return argparse.Namespace(
        checkpoint=str(checkpoint),
        out_dir=str(tmp_path / "sidecar"),
        data_dir="",
        data=[str(data)],
        preset="omnicoder2026_full_ledger_probe",
        rank_device_map="cpu",
        placement_layer_counts="1",
        precision="fp32",
        init_dtype="fp32",
        nproc_per_node=1,
        seq_len=16,
        max_records_per_file=1,
        decode_max_prompt_tokens=128,
        decode_max_output_tokens=16,
        fake_quant=False,
        fake_quant_chunk_rows=0,
        fake_quant_max_full_elements=0,
        require_target_contract=False,
        timeout_seconds=10,
        dry_run=True,
    )


def test_checkpoint_eval_sidecar_dry_run_plans_decode_and_diagnostics(tmp_path) -> None:
    manifest = sidecar.run_sidecar(_args(tmp_path))
    out_dir = tmp_path / "sidecar"
    written = json.loads((out_dir / "checkpoint_eval_manifest.json").read_text(encoding="utf-8"))
    job_names = {job["name"] for job in manifest["jobs"]}

    assert manifest["status"] == "planned"
    assert written["schema"] == sidecar.SCHEMA
    assert "media_route_probe" in job_names
    assert "target_token_diagnostics" in job_names
    assert "heldout_pipeline_sample_loss" in job_names
    assert "token_topk_probe" in job_names
    assert "decode_sanity_predictions" in job_names
    assert (out_dir / "decode_sanity_tasks.jsonl").exists()
    decode_job = next(job for job in manifest["jobs"] if job["name"] == "decode_sanity_predictions")
    assert "--allow-local-dev-tasks" in decode_job["cmd"]
    assert "--max-output-tokens" in decode_job["cmd"]


def test_checkpoint_eval_sidecar_records_timeout_failures(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    args.dry_run = False

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(sidecar.subprocess, "run", raise_timeout)
    manifest = sidecar.run_sidecar(args)

    assert manifest["status"] == "failed"
    assert (tmp_path / "sidecar" / "checkpoint_eval_manifest.json").exists()
    failed = [job for job in manifest["jobs"] if job["status"] == "failed"]
    assert failed
    assert any("timeout_expired" in str(job.get("error", "")) for job in failed)
