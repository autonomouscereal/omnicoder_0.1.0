from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from omnicoder.eval.sample_loss_2026 import load_native_checkpoint
from omnicoder.training.pretrain_2026_dense import _text_from_record
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def iter_jsonl(path: str | Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    seen = 0
    source = Path(path)
    if not source.exists():
        return
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if limit and seen >= limit:
            break
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            payload = {"text": line}
        if isinstance(payload, dict):
            seen += 1
            yield payload


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def compact_json(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return text[:limit]


def reward_value(record: dict[str, Any]) -> float:
    for key in ("reward", "score"):
        value = record.get(key)
        if value is not None:
            try:
                return max(-1.0, min(1.0, float(value)))
            except Exception:
                pass
    verifier = record.get("verifier")
    if isinstance(verifier, dict) and verifier.get("reward") is not None:
        try:
            return max(-1.0, min(1.0, float(verifier["reward"])))
        except Exception:
            pass
    quality = record.get("quality_score")
    try:
        return max(-1.0, min(1.0, float(quality)))
    except Exception:
        return 0.5


def record_to_text_and_weight(record: dict[str, Any]) -> tuple[str, float]:
    kind = str(record.get("training_kind") or "")
    prompt = str(record.get("prompt") or "")
    reward = reward_value(record)
    if kind == "tool_preference" or {"prompt", "chosen", "rejected"} <= set(record):
        text = f"user: {prompt}\nassistant: {record.get('chosen', '')}"
        weight = 1.25 + max(0.0, reward) * 0.5
    elif kind == "tool_reward":
        target = {
            "tool_calls": record.get("tool_calls", []),
            "tool_results": record.get("tool_results", []),
            "reward": reward,
            "reward_components": record.get("reward_components", {}),
        }
        text = f"user: {prompt}\nassistant: {compact_json(target)}"
        weight = 0.5 + max(0.0, reward) * 1.5
    elif kind == "tool_rlvr":
        target = {
            "verifier": record.get("verifier", {}),
            "environment": record.get("environment", {}),
            "tool_calls": record.get("tool_calls", []),
        }
        text = f"user: {prompt}\nassistant: {compact_json(target)}"
        weight = 0.75 + max(0.0, reward) * 1.25
    elif kind == "tool_safety_negative":
        text = f"user: {prompt}\nassistant: {record.get('chosen', 'Refuse unsafe tool use and protect credentials.')}"
        weight = 1.5
    else:
        text = _text_from_record(record)
        weight = 1.0 + max(0.0, reward) * 0.25
    text = text.strip()
    return text, max(0.05, min(2.5, float(weight)))


class RewardReplayDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], tokenizer: Any, seq_len: int, max_records: int = 0):
        self.samples: list[tuple[list[int], float, str]] = []
        per_file_limit = int(max_records) if int(max_records) > 0 and len(paths) == 1 else 0
        remaining = int(max_records) if int(max_records) > 0 else 0
        for path in paths:
            limit = per_file_limit or remaining
            for record in iter_jsonl(path, limit):
                if remaining and len(self.samples) >= remaining:
                    break
                text, weight = record_to_text_and_weight(record)
                if not text:
                    continue
                ids = [int(item) for item in tokenizer.encode(text)]
                for start in range(0, len(ids), max(1, seq_len)):
                    chunk = ids[start:start + seq_len]
                    if len(chunk) >= 2:
                        self.samples.append((chunk, weight, stable_hash({"path": path, "record": record})[:16]))
                        break
            if remaining and len(self.samples) >= remaining:
                break
        if not self.samples:
            self.samples.append(([1, 1], 0.05, "empty_fallback"))
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        ids, weight, sample_id = self.samples[idx]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        return torch.tensor(ids[: self.seq_len], dtype=torch.long), torch.tensor(float(weight), dtype=torch.float32), sample_id


def weighted_ce(logits: torch.Tensor, batch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(batch.shape[0], -1)
    mask = (batch[:, 1:] != 0).float()
    per_sample = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return (per_sample * weights.to(per_sample.device)).mean()


def save_checkpoint(out: str | Path, source_checkpoint: str | Path, model: torch.nn.Module, opt: torch.optim.Optimizer, args: argparse.Namespace, last_loss: float | None) -> None:
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        source = {}
    source["model_state_dict"] = model.state_dict()
    source["optimizer_state_dict"] = opt.state_dict()
    source["reward_replay"] = {
        "schema": "omnicoder.reward_replay_2026.v1",
        "train_jsonl": args.train_jsonl,
        "steps": int(args.steps),
        "seq_len": int(args.seq_len),
        "learning_rate": float(args.lr),
        "max_records": int(args.max_records),
        "last_loss": last_loss,
    }
    source.setdefault("train_args", {})
    if isinstance(source["train_args"], dict):
        source["train_args"]["reward_replay_2026"] = True
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reward/preference/RLVR replay for native Omnicoder2026 checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-jsonl", action="append", default=[], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="ledger_probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    tokenizer = get_text_tokenizer(prefer_hf=True)
    model = load_native_checkpoint(args.checkpoint, args.profile, device)
    model.train()
    dataset = RewardReplayDataset(args.train_jsonl, tokenizer, int(args.seq_len), int(args.max_records))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    last_loss: float | None = None
    iterator = iter(loader)
    for step in range(1, int(args.steps) + 1):
        try:
            batch, weights, sample_ids = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, weights, sample_ids = next(iterator)
        batch = batch.to(device)
        weights = weights.to(device)
        out = model(batch)
        loss = weighted_ce(out["logits"], batch, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        event = {
            "step": step,
            "loss": last_loss,
            "weight_mean": float(weights.detach().mean().cpu()),
            "sample_ids": list(sample_ids),
            "dataset_records": len(dataset),
        }
        line = json.dumps(event, ensure_ascii=True, sort_keys=True)
        print(line)
        if log_path:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    save_checkpoint(args.out, args.checkpoint, model, opt, args, last_loss)
    done = {"status": "ok", "out": args.out, "last_loss": last_loss, "dataset_records": len(dataset)}
    print(json.dumps(done, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
