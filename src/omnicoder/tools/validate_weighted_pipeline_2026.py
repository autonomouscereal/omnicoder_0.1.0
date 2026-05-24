from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config


def tiny_config(pattern: tuple[str, str]) -> OmniCoder2026Config:
    return OmniCoder2026Config(
        vocab_size=96,
        n_layers=2,
        d_model=32,
        n_heads=4,
        head_dim=8,
        num_key_value_heads=1,
        mlp_dim=64,
        max_seq_len=32,
        local_window=16,
        csa_block_size=16,
        csa_top_k_blocks=2,
        hca_block_size=16,
        latent_dim=8,
        rope_dim=8,
        sink_tokens=1,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        index_head_dim=8,
        flow_latent_dim=8,
        layer_pattern=pattern,  # type: ignore[arg-type]
    )


def parse_devices(raw: str) -> list[torch.device]:
    devices: list[torch.device] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item == "cpu" or item.startswith("cpu"):
            devices.append(torch.device(item))
        elif item.startswith("cuda"):
            devices.append(torch.device(item))
        else:
            devices.append(torch.device("cuda", int(item)))
    if len(devices) != 2:
        raise ValueError("--devices must resolve to exactly two devices for this validator")
    return devices


def make_model(cfg: OmniCoder2026Config, devices: list[torch.device]) -> OmniCoder2026:
    torch.manual_seed(11)
    model = OmniCoder2026(cfg)
    model.apply_weighted_device_map(
        devices,
        embed_device=devices[-1],
        head_device=devices[-1],
    )
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate weighted-placement microbatch scheduler parity.")
    parser.add_argument("--devices", default="cpu,cpu")
    parser.add_argument("--pattern", default="kda,kda")
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-5)
    parser.add_argument("--enable-async", action="store_true", help="Also remove the async hard gate and validate real CUDA streams.")
    args = parser.parse_args(argv)

    devices = parse_devices(args.devices)
    pattern_items = tuple(part.strip() for part in str(args.pattern).split(",") if part.strip())
    if len(pattern_items) != 2:
        raise ValueError("--pattern must contain exactly two block kinds")
    cfg = tiny_config(pattern_items)  # type: ignore[arg-type]
    batch = torch.tensor(
        [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17]],
        dtype=torch.long,
    )
    if args.enable_async:
        os.environ["OMNICODER2026_ENABLE_ASYNC_PIPELINE"] = "1"
    else:
        os.environ.pop("OMNICODER2026_ENABLE_ASYNC_PIPELINE", None)

    losses: dict[str, float] = {}
    for mode in ("standard", "serial", "async_request"):
        model = make_model(cfg, devices)
        if mode == "standard":
            loss = model(batch, labels=batch, return_logits=False, return_hidden=False)["loss"]
        elif mode == "serial":
            loss = model.forward_weighted_pipeline_loss(batch, batch, microbatches=args.microbatches, async_streams=False)
        else:
            loss = model.forward_weighted_pipeline_loss(batch, batch, microbatches=args.microbatches, async_streams=True)
        if loss is None:
            raise RuntimeError(f"{mode} did not return a loss")
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        losses[mode] = float(loss.detach().cpu())

    baseline = losses["standard"]
    comparisons = {
        key: {
            "loss": value,
            "abs_delta": abs(value - baseline),
            "passed": abs(value - baseline) <= float(args.atol) + float(args.rtol) * abs(baseline),
        }
        for key, value in losses.items()
        if key != "standard"
    }
    result = {
        "status": "passed" if all(item["passed"] for item in comparisons.values()) else "failed",
        "devices": [str(device) for device in devices],
        "device_names": [
            torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else str(device)
            for device in devices
        ],
        "pattern": list(pattern_items),
        "async_enabled": bool(args.enable_async),
        "losses": losses,
        "comparisons": comparisons,
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
