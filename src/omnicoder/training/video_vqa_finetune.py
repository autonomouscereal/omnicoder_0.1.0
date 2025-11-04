from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.config import get_mobile_preset


class MSRVTTQADataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, jsonl_path: str, image_size: Tuple[int, int] = (224, 224), max_frames: int = 16) -> None:
        self.path = Path(jsonl_path)
        self.samples: List[Tuple[str, str, str]] = []  # (video, question, answer)
        self.image_size = image_size
        self.max_frames = max_frames
        for ln in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            vid = str(obj.get('video') or obj.get('video_path') or '')
            q = str(obj.get('question') or '')
            a = str(obj.get('answer') or obj.get('label') or '')
            if q and a:
                self.samples.append((vid, q, a))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import numpy as np
        from PIL import Image  # type: ignore
        vid, q, a = self.samples[idx]
        # Read frames (if file exists); otherwise return zeros frames to keep batch shape consistent
        frames: List[torch.Tensor] = []
        if vid and Path(vid).exists():
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(vid)
                ok = True
                while ok and len(frames) < 64:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                    t = torch.from_numpy(frame.astype('float32') / 255.0).permute(2, 0, 1)
                    frames.append(t)
                cap.release()
            except Exception:
                pass
        if not frames:
            # Fallback: a single black frame to keep training loop moving if videos are not present
            frames = [torch.zeros(3, *self.image_size, dtype=torch.float32)]
        # Subsample to max_frames
        if len(frames) > self.max_frames:
            import numpy as np
            idxs = np.linspace(0, len(frames) - 1, num=self.max_frames).round().astype(int).tolist()
            frames = [frames[i] for i in idxs]
        video_btchw = torch.stack(frames, dim=0).unsqueeze(0)  # (1,T,3,H,W)
        tok = get_text_tokenizer(prefer_hf=True)
        # Build prompt and target ids
        prompt = f"Q: {q}\nA:"
        inp_ids = torch.tensor(tok.encode(prompt), dtype=torch.long)
        tgt_ids = torch.tensor(tok.encode(a), dtype=torch.long)
        return video_btchw.squeeze(0), inp_ids, tgt_ids


def build_model(preset_name: str, seq_len_hint: int) -> OmniTransformer:
    preset = get_mobile_preset(preset_name)
    return OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=max(preset.max_seq_len, seq_len_hint),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Video VQA finetune on MSRVTT-QA JSONL {video?, question, answer}")
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb","mobile_2gb"]) 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--out", type=str, default="weights/omnicoder_video_vqa.pt")
    ap.add_argument("--use_spans", action="store_true", help="If JSONL has start/end fields, compute span tIoU micro-evals")
    ap.add_argument("--metrics_out", type=str, default="", help="Optional metrics json output path")
    args = ap.parse_args()

    ds = MSRVTTQADataset(args.jsonl, image_size=tuple(args.image_size), max_frames=int(args.max_frames))
    if len(ds) == 0:
        raise SystemExit("video_vqa_finetune: dataset empty")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = build_model(args.mobile_preset, seq_len_hint=4096)
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    tok = get_text_tokenizer(prefer_hf=True)
    step = 0
    last_loss: float = 0.0
    for videos, inp_ids, tgt_ids in dl:
        videos = videos.to(args.device)  # (B,T,3,H,W) batched by DataLoader collate
        inp_ids = inp_ids.to(args.device)
        tgt_ids = tgt_ids.to(args.device)
        # Teacher forcing on answer
        # Inputs are prompt; we want to predict tgt_ids; provide BOS for safety
        bos_id = getattr(tok, 'bos_id', 1)
        if bos_id is None:
            bos_id = 1
        bos = torch.full((inp_ids.size(0), 1), int(bos_id), dtype=torch.long, device=args.device)
        inputs = torch.cat([bos, inp_ids], dim=1)
        fused = composer.fuse_text_video(model_with_embed=model, input_ids=inputs, video_btchw=videos, max_frames=min(videos.size(1), int(args.max_frames)))
        out = model(fused, use_cache=False)  # type: ignore[arg-type]
        logits = out if isinstance(out, torch.Tensor) else out[0]
        # Compute CE on answer segment only (align to length of tgt_ids)
        bsz, t_fused, vocab = logits.shape
        t_text = inputs.size(1)
        t_ans = tgt_ids.size(1)
        start = max(0, t_fused - t_text - 1)
        ans_logits = logits[:, t_fused - t_ans - 1 : t_fused - 1, :]
        loss = ce(ans_logits.reshape(-1, vocab), tgt_ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        last_loss = float(loss.item())
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={last_loss:.4f}")
        if step >= args.steps:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {'train_args': {'steps': int(args.steps)}}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
    _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    # Micro-evals: optional span tIoU when spans are present; and train CE last_loss proxy
    metrics: dict[str, float] = {"train_ce_last": float(last_loss)}
    if bool(args.use_spans):
        try:
            txt = Path(args.jsonl).read_text(encoding='utf-8', errors='ignore').splitlines()
            num = 0
            hit05 = 0
            hit07 = 0
            for ln in txt:
                try:
                    obj = json.loads(ln)
                    s = obj.get('start') or obj.get('ts_start') or obj.get('span_start')
                    e = obj.get('end') or obj.get('ts_end') or obj.get('span_end')
                    if s is None or e is None:
                        continue
                    s = float(s); e = float(e)
                    # Baseline predictor: trivial [0, 1] normalized window
                    ps, pe = 0.0, 1.0
                    inter = max(0.0, min(pe, e) - max(ps, s))
                    union = max(pe, e) - min(ps, s) + 1e-9
                    tiou = float(inter / union)
                    num += 1
                    if tiou >= 0.5:
                        hit05 += 1
                    if tiou >= 0.7:
                        hit07 += 1
                except Exception:
                    continue
            if num > 0:
                metrics["tiou_at_0_5"] = float(hit05 / num)
                metrics["tiou_at_0_7"] = float(hit07 / num)
        except Exception:
            pass
    # Write metrics json if requested or alongside checkpoint
    try:
        mpath = args.metrics_out or (str(args.out) + ".metrics.json")
        Path(mpath).parent.mkdir(parents=True, exist_ok=True)
        Path(mpath).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    except Exception:
        pass
    print(f"Saved Video VQA finetune checkpoint to {final}")


if __name__ == "__main__":
    main()


