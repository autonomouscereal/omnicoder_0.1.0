from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config
from omnicoder.tokenization.native_media_2026 import (
    NativeMediaPacket,
    image_to_native_patches,
    ocr_image_to_native_patches,
    video_to_native_patches,
    waveform_to_native_segments,
)

SCHEMA = "omnicoder.native_media_trainability_2026.v1"
CORE_MODALITIES = ("image", "video", "audio", "music", "tts", "ocr")


def _tiny_config(
    *,
    vocab_size: int = 4096,
    d_model: int = 128,
    feature_dim: int = 64,
    max_seq_len: int = 128,
) -> OmniCoder2026Config:
    return OmniCoder2026Config(
        vocab_size=int(vocab_size),
        n_layers=3,
        d_model=int(d_model),
        n_heads=4,
        head_dim=max(16, int(d_model) // 4),
        num_key_value_heads=1,
        mlp_dim=int(d_model) * 4,
        max_seq_len=int(max_seq_len),
        local_window=64,
        csa_block_size=32,
        csa_top_k_blocks=4,
        hca_block_size=64,
        latent_dim=max(16, int(d_model) // 4),
        rope_dim=16,
        sink_tokens=1,
        q_lora_rank=max(16, int(d_model) // 2),
        o_lora_rank=max(16, int(d_model) // 2),
        o_groups=1,
        csa_compress_rate=2,
        hca_compress_rate=4,
        index_head_dim=16,
        hc_mult=1,
        residual_mode="block_attnres",
        block_attnres_block_size=16,
        block_attnres_max_blocks=8,
        block_attnres_rank=16,
        block_attnres_chunk_tokens=32,
        layer_pattern=("kda", "csa", "hca"),
        tie_embeddings=False,
        mtp_heads=1,
        flow_latent_dim=max(32, int(d_model) // 2),
        native_media_feature_dim=int(feature_dim),
        native_media_position_dim=4,
        native_media_type_vocab=16,
        fake_quant=False,
    )


def _synthetic_packets(cfg: OmniCoder2026Config, *, device: torch.device, seed: int) -> dict[str, NativeMediaPacket]:
    gen = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    gen.manual_seed(int(seed))
    feature_dim = int(cfg.native_media_feature_dim)

    image = torch.randn((1, 3, 32, 32), generator=gen, device=device)
    video = torch.randn((1, 3, 3, 24, 24), generator=gen, device=device)
    audio = torch.randn((1, 256), generator=gen, device=device)
    music = torch.sin(torch.linspace(0, 18.0, 384, device=device)).view(1, -1)
    tts = torch.cos(torch.linspace(0, 9.0, 256, device=device)).view(1, -1)
    ocr = torch.randn((1, 3, 32, 48), generator=gen, device=device)

    return {
        "image": image_to_native_patches(image, patch=16, feature_dim=feature_dim),
        "video": video_to_native_patches(video, patch=12, feature_dim=feature_dim),
        "audio": waveform_to_native_segments(audio, kind="audio", segment=64, stride=64, feature_dim=feature_dim),
        "music": waveform_to_native_segments(music, kind="music", segment=64, stride=64, feature_dim=feature_dim),
        "tts": waveform_to_native_segments(tts, kind="tts", segment=64, stride=64, feature_dim=feature_dim),
        "ocr": ocr_image_to_native_patches(ocr, patch=16, feature_dim=feature_dim),
    }


def _aligned_input_ids(packet: NativeMediaPacket, cfg: OmniCoder2026Config, *, device: torch.device) -> torch.Tensor:
    length = max(2, int(packet.features.shape[1]) + 1)
    ids = torch.zeros((1, min(length, int(cfg.max_seq_len))), dtype=torch.long, device=device)
    return ids


def _packet_loss(model: OmniCoder2026, cfg: OmniCoder2026Config, packet: NativeMediaPacket, *, device: torch.device) -> torch.Tensor:
    token_count = min(int(packet.features.shape[1]), int(cfg.max_seq_len) - 1)
    features = packet.features[:, :token_count, :].to(device)
    targets = features.detach().clone()
    input_ids = _aligned_input_ids(packet, cfg, device=device)
    result = model(
        input_ids=input_ids,
        native_media_features=features,
        native_media_type_ids=packet.type_ids[:, :token_count].to(device),
        native_media_positions=packet.positions[:, :token_count, :].to(device),
        native_media_targets=targets,
        native_media_mask=torch.ones((1, token_count), dtype=torch.float32, device=device),
        return_logits=False,
        return_hidden=False,
    )
    loss = result.get("native_media_loss")
    if loss is None:
        raise RuntimeError(f"model did not return native_media_loss for {packet.kind}")
    return loss


@torch.no_grad()
def _loss_snapshot(
    model: OmniCoder2026,
    cfg: OmniCoder2026Config,
    packets: dict[str, NativeMediaPacket],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for modality in CORE_MODALITIES:
        loss = _packet_loss(model, cfg, packets[modality], device=device)
        losses[modality] = float(loss.detach().cpu())
    return losses


def run_native_media_trainability_probe(
    *,
    steps: int = 40,
    learning_rate: float = 3e-4,
    min_loss_drop_ratio: float = 0.08,
    device: str = "auto",
    seed: int = 2026,
) -> dict[str, Any]:
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    torch.manual_seed(int(seed))
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    cfg = _tiny_config()
    model = OmniCoder2026(cfg).to(selected_device)
    packets = _synthetic_packets(cfg, device=selected_device, seed=int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)

    initial = _loss_snapshot(model, cfg, packets, device=selected_device)
    trace: list[dict[str, Any]] = []
    for step in range(1, max(1, int(steps)) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = [_packet_loss(model, cfg, packets[modality], device=selected_device) for modality in CORE_MODALITIES]
        total = torch.stack(losses).mean()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step == int(steps) or step % max(1, int(steps) // 4) == 0:
            trace.append(
                {
                    "step": int(step),
                    "loss": float(total.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                }
            )

    final = _loss_snapshot(model, cfg, packets, device=selected_device)
    ratios: dict[str, float] = {}
    passed_modalities: dict[str, bool] = {}
    for modality in CORE_MODALITIES:
        before = max(float(initial[modality]), 1e-12)
        after = float(final[modality])
        ratios[modality] = (before - after) / before
        passed_modalities[modality] = bool(after < before and ratios[modality] >= float(min_loss_drop_ratio))

    status = "passed" if all(passed_modalities.values()) else "failed"
    return {
        "schema": SCHEMA,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(selected_device),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "min_loss_drop_ratio": float(min_loss_drop_ratio),
        "model": {
            "n_layers": int(cfg.n_layers),
            "d_model": int(cfg.d_model),
            "native_media_feature_dim": int(cfg.native_media_feature_dim),
            "residual_mode": str(cfg.residual_mode),
            "layer_pattern": list(cfg.layer_pattern),
        },
        "initial_loss": initial,
        "final_loss": final,
        "loss_drop_ratio": ratios,
        "passed_modalities": passed_modalities,
        "trace": trace,
        "contract": {
            "continuous_media_path": "one shared NativeContinuousMediaBridge for image/video/audio/music/tts/ocr",
            "edge_policy": "patchify or segmentize only; no modality-specific learned encoder inside the trunk",
            "training_meaning": "scratch shared trunk receives gradients from every core media modality and reduces reconstruction loss",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native continuous media trainability proof for Omnicoder 2026")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--min-loss-drop-ratio", "--min_loss_drop_ratio", dest="min_loss_drop_ratio", type=float, default=0.08)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = run_native_media_trainability_probe(
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        min_loss_drop_ratio=float(args.min_loss_drop_ratio),
        device=str(args.device),
        seed=int(args.seed),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(out), "loss_drop_ratio": report["loss_drop_ratio"]}, ensure_ascii=True), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
