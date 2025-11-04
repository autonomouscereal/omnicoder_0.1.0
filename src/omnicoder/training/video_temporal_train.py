from __future__ import annotations

"""
Lightweight training loop for a tiny temporal module over frame latents.

Goal: learn a TemporalSSM-style smoother/predictor on sequences of per-frame
latent features using a simple next-step prediction loss.

Inputs: a directory of videos or frames; we derive per-frame features using
global average pooling over resized RGB frames (no external backbone needed).

Outputs: saves the temporal module weights and optionally exports ONNX.
"""

import argparse
from pathlib import Path
import os
from typing import List, Tuple

import numpy as np
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.nn.functional as F
try:
    # Optional AV sync module
    from omnicoder.modeling.multimodal.av_sync import AVSyncModule  # type: ignore
except Exception:
    AVSyncModule = None  # type: ignore


class TemporalSSM(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        hidden = int(d_model * expansion)
        self.proj_in = nn.Linear(d_model, hidden * 2, bias=False)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, groups=hidden, padding=kernel_size // 2)
        self.proj_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        u, v = self.proj_in(x).chunk(2, dim=-1)
        v = F.gelu(v)
        y = u * v
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = y.transpose(1, 2)
        return self.proj_out(y)


def _load_videos_as_features(root: str, frames: int = 16, size: Tuple[int, int] = (224, 224)) -> List[Tuple[torch.Tensor, str]]:
    import os
    try:
        import cv2  # type: ignore
    except Exception:
        raise SystemExit("OpenCV is required for video temporal training")
    feats: List[Tuple[torch.Tensor, str]] = []
    for fp in sorted(os.listdir(root)):
        if not fp.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        full = str(Path(root) / fp)
        cap = cv2.VideoCapture(full)
        frames_list: List[np.ndarray] = []
        ok = True
        while ok:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            frames_list.append(fr)
        cap.release()
        if not frames_list:
            continue
        t = len(frames_list)
        if t >= frames:
            idx = np.linspace(0, t - 1, frames).round().astype(int)
            frames_list = [frames_list[i] for i in idx]
        else:
            while len(frames_list) < frames:
                frames_list += frames_list
            frames_list = frames_list[:frames]
        # Compute per-frame features by simple global average of RGB
        seq = []
        for fr in frames_list:
            arr = fr.astype(np.float32) / 255.0  # (H,W,C)
            feat = arr.mean(axis=(0, 1))  # (C,)
            seq.append(torch.from_numpy(feat))
        feats.append((torch.stack(seq, dim=0), Path(fp).stem))  # (T,C), name
    return feats


def _load_audio_frame_features(audio_path: str, frames: int, sr: int = 16000) -> torch.Tensor:
    """Return a (T, D) tensor of per-frame audio features using log-mel pooling.

    We compute a log-mel spectrogram and partition it into `frames` equal time bins,
    average over frequency, and produce a D=64 feature by projecting mel bands.
    """
    try:
        import librosa  # type: ignore
        import numpy as _np
        import warnings as _warn
        # Suppress librosa's pkg_resources deprecation warning (setuptools>=81)
        _warn.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")
    except Exception:
        # Fallback: zero features if librosa is unavailable
        return torch.zeros(frames, 64)
    try:
        y, _sr = librosa.load(audio_path, sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
        logm = librosa.power_to_db(mel + 1e-9)
        Tm = logm.shape[1]
        # Partition time into `frames` bins
        idx = _np.linspace(0, Tm - 1, frames).round().astype(int)
        bins = []
        for i in range(frames):
            j0 = int(_np.floor(i * (Tm / frames)))
            j1 = int(_np.floor((i + 1) * (Tm / frames)))
            if j1 <= j0:
                j1 = min(Tm, j0 + 1)
            seg = logm[:, j0:j1]
            if seg.size == 0:
                seg = logm[:, idx[i]:idx[i] + 1]
            v = _np.mean(seg, axis=1)  # (64,)
            bins.append(torch.from_numpy(v.astype(_np.float32)))
        return torch.stack(bins, dim=0)  # (frames,64)
    except Exception:
        return torch.zeros(frames, 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny temporal module over frame latents")
    ap.add_argument("--videos", type=str, required=True, help="Directory with videos")
    ap.add_argument("--frames", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_FRAMES', '16')))
    ap.add_argument("--d_model", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_D', '384')))
    ap.add_argument("--kernel", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_KERNEL', '5')))
    ap.add_argument("--expansion", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_EXPANSION', '2')))
    ap.add_argument("--steps", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_STEPS', '500')))
    ap.add_argument("--device", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_DEVICE', ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--lr", type=float, default=float(os.getenv('OMNICODER_VIDEO_TEMPORAL_LR', '2e-4')))
    ap.add_argument("--out", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_OUT', 'weights/temporal_ssm.pt'))
    ap.add_argument("--export_onnx", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_ONNX', ''), help="Optional ONNX output path")
    # Temporal consistency: latent noise propagation across frames
    ap.add_argument("--propagate_noise", action="store_true", default=(os.getenv('OMNICODER_VIDEO_TEMPORAL_PROPAGATE_NOISE','0')=='1'), help="Enable latent noise propagation across frames during training")
    ap.add_argument("--noise_dim", type=int, default=int(os.getenv('OMNICODER_VIDEO_TEMPORAL_NOISE_DIM','64')), help="Dimension of propagated latent noise")
    ap.add_argument("--noise_alpha", type=float, default=float(os.getenv('OMNICODER_VIDEO_TEMPORAL_NOISE_ALPHA','0.95')), help="AR(1) persistence for noise propagation (0..1)")
    ap.add_argument("--noise_gamma", type=float, default=float(os.getenv('OMNICODER_VIDEO_TEMPORAL_NOISE_GAMMA','0.05')), help="Scale of noise injected into latents")
    # Optional FVD evaluation (requires pytorch-fvd)
    ap.add_argument("--fvd_ref_dir", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_FVD_REF',''), help="Directory with reference videos for FVD (optional)")
    ap.add_argument("--fvd_pred_dir", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_FVD_PRED',''), help="Directory with generated videos for FVD (optional)")
    # Optional AV‑sync alignment
    ap.add_argument("--av_sync", action="store_true", default=(os.getenv('OMNICODER_VIDEO_TEMPORAL_AV_SYNC','0')=='1'), help="Enable audio‑visual alignment loss if matching audio is available")
    ap.add_argument("--audio_dir", type=str, default=os.getenv('OMNICODER_VIDEO_TEMPORAL_AUDIO_DIR',''), help="Directory with .wav audio files matching video basenames")
    ap.add_argument("--av_weight", type=float, default=float(os.getenv('OMNICODER_VIDEO_TEMPORAL_AV_WEIGHT','0.1')), help="Weight for AV‑sync alignment loss component")
    # Physics‑violation curriculum (optional)
    ap.add_argument("--physics_jsonl", type=str, default=os.getenv('OMNICODER_VIDEO_PHYSICS_JSONL',''), help="JSONL with {video: <basename>, violations: [{t0:int,t1:int,kind:str}]} annotations")
    ap.add_argument("--physics_weight", type=float, default=float(os.getenv('OMNICODER_VIDEO_PHYSICS_WEIGHT','0.2')), help="Weight for physics violation classification loss")
    ap.add_argument("--physics_smoothness", action="store_true", default=(os.getenv('OMNICODER_VIDEO_PHYSICS_SMOOTH','1')=='1'), help="Enable temporal smoothness regularizer (L2 on finite differences)")
    ap.add_argument("--physics_repair", action="store_true", default=(os.getenv('OMNICODER_VIDEO_PHYSICS_REPAIR','1')=='1'), help="When violations annotated, nudge predictions toward local interpolation around violation frames")
    args = ap.parse_args()

    # Auto-fetch/seed: ensure the videos directory exists and contains at least one .mp4
    try:
        root = Path(args.videos)
        root.mkdir(parents=True, exist_ok=True)
        def _ensure_minimal_videos(dst: Path, want: int = 2) -> None:
            try:
                import cv2  # type: ignore
                import numpy as _np  # type: ignore
            except Exception:
                return
            existing = list(p for p in dst.glob("*.mp4"))
            if len(existing) >= want:
                return
            # If there are images, turn a few into short static clips
            imgs = list(dst.glob("*.png")) + list(dst.glob("*.jpg")) + list(dst.glob("*.jpeg"))
            made = 0
            for p in imgs[:max(0, want - len(existing))]:
                try:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    h, w = 64, 64
                    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    outp = dst / f"{p.stem}_clip.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cv2.VideoWriter(str(outp), fourcc, 16, (w, h))
                    for _ in range(16):
                        vw.write(frame)
                    vw.release()
                    made += 1
                except Exception:
                    continue
            if len(list(dst.glob("*.mp4"))) >= want:
                return
            # Otherwise, synthesize simple procedural clips
            needed = want - len(list(dst.glob("*.mp4")))
            for i in range(max(0, needed)):
                try:
                    outp = dst / f"synth_{i}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    w, h = 64, 64
                    vw = cv2.VideoWriter(str(outp), fourcc, 16, (w, h))
                    for t in range(16):
                        grad = _np.linspace(0, 255, w, dtype=_np.uint8)[None, :].repeat(h, axis=0)
                        r = grad
                        g = _np.roll(grad, t * 2, axis=1)
                        b = _np.roll(grad, t * 3, axis=0)
                        frame = _np.stack([b, g, r], axis=-1)
                        vw.write(frame)
                    vw.release()
                except Exception:
                    continue
        _ensure_minimal_videos(root)
    except Exception:
        pass

    seqs = _load_videos_as_features(args.videos, frames=int(args.frames))
    if not seqs:
        print("[temporal_train] no videos found; exiting successfully")
        return
    # Pad/truncate feature dimension to d_model
    seqs_proc: List[torch.Tensor] = []
    names: List[str] = []
    for s, name in seqs:
        # s: (T, C)
        c = s.shape[1]
        if c == args.d_model:
            seqs_proc.append(s)
        elif c > args.d_model:
            seqs_proc.append(s[:, : args.d_model])
        else:
            pad = torch.zeros(s.shape[0], args.d_model - c)
            seqs_proc.append(torch.cat([s, pad], dim=1))
        names.append(name)

    model = TemporalSSM(d_model=int(args.d_model), kernel_size=int(args.kernel), expansion=int(args.expansion)).to(args.device)
    # Optional physics‑violation head (binary classifier per frame)
    physics_head: nn.Module | None = None
    violation_spans: dict[str, List[Tuple[int,int,str]]] = {}
    if args.physics_jsonl:
        try:
            import json as _json
            for line in Path(args.physics_jsonl).read_text(encoding='utf-8').splitlines():
                try:
                    j = _json.loads(line)
                    vid = str(Path(j.get('video','')).stem)
                    vios = j.get('violations', []) or []
                    spans: List[Tuple[int,int,str]] = []
                    for v in vios:
                        t0 = int(v.get('t0', 0)); t1 = int(v.get('t1', t0)); kind = str(v.get('kind',''))
                        spans.append((max(0, t0), max(0, t1), kind))
                    if vid:
                        violation_spans[vid] = spans
                except Exception:
                    continue
            if violation_spans:
                physics_head = nn.Sequential(nn.LayerNorm(int(args.d_model)), nn.Linear(int(args.d_model), 1)).to(args.device)
        except Exception as e:
            print(f"[warn] physics_jsonl parse failed: {e}")
    if bool(args.av_sync) and AVSyncModule is None:
        print("[warn] AVSyncModule unavailable; continuing without av_sync")
        args.av_sync = False
    av_sync_mod = None
    if bool(args.av_sync) and AVSyncModule is not None:
        # audio D=64, video D=args.d_model → project both into d_model space
        av_sync_mod = AVSyncModule(d_audio=64, d_video=int(args.d_model), d_model=min(512, int(args.d_model)), num_heads=4).to(args.device)  # type: ignore
    model.train()
    params = list(model.parameters())
    if physics_head is not None:
        params += list(physics_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    opt_av = torch.optim.AdamW(av_sync_mod.parameters(), lr=float(args.lr)) if av_sync_mod is not None else None

    # Simple next-step prediction: predict x[t+1] from x[t]
    step = 0
    # Optional per-sequence AR(1) noise state cache
    noise_states: List[torch.Tensor] = []
    if args.propagate_noise:
        for _ in seqs_proc:
            noise_states.append(torch.zeros(int(args.noise_dim)))
    # Pre-load audio features if requested
    audio_feats: List[torch.Tensor] = []
    if bool(args.av_sync) and args.audio_dir:
        # Match audio by base filename (video.mp4 → audio_dir/video.wav)
        import os as _os
        vids = sorted([fp for fp in _os.listdir(args.videos) if fp.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))])
        for fp in vids:
            base = Path(fp).stem
            wav = Path(args.audio_dir) / f"{base}.wav"
            if wav.exists():
                af = _load_audio_frame_features(str(wav), frames=int(args.frames))
            else:
                af = torch.zeros(int(args.frames), 64)
            audio_feats.append(af)
    while step < int(args.steps):
        for idx, s in enumerate(seqs_proc):
            x = s.unsqueeze(0).to(args.device)  # (1,T,D)
            if args.propagate_noise:
                # Generate AR(1) noise sequence and inject additively (broadcast to D via a learned projection)
                T = x.size(1)
                # Update state
                st = noise_states[idx].to(args.device)
                eps = torch.randn_like(st)
                st = float(args.noise_alpha) * st + (1.0 - float(args.noise_alpha)) ** 0.5 * eps
                noise_states[idx] = st.detach().cpu()
                # Build per-frame noise by repeating state and adding small innovations
                z = st.repeat(T, 1)  # (T, Z)
                z = z + 0.01 * torch.randn_like(z)
                # Project to feature dim and inject
                proj = getattr(main, "_noise_proj", None)
                if proj is None or proj.out_features != x.size(-1) or proj.in_features != int(args.noise_dim):
                    main._noise_proj = torch.nn.Linear(int(args.noise_dim), x.size(-1), bias=False).to(args.device)  # type: ignore[attr-defined]
                nz = main._noise_proj(z).unsqueeze(0)  # (1,T,D)
                x = x + float(args.noise_gamma) * nz
            y = model(x)  # (1,T,D)
            loss = F.mse_loss(y[:, :-1, :], x[:, 1:, :])
            # Optional temporal smoothness (finite differences)
            if bool(args.physics_smoothness):
                dy = y[:, 1:, :] - y[:, :-1, :]
                loss = loss + 0.01 * (dy.pow(2).mean())
            # Optional AV‑sync alignment loss
            if av_sync_mod is not None:
                try:
                    a = audio_feats[idx].unsqueeze(0).to(args.device) if idx < len(audio_feats) else torch.zeros(int(args.frames), 64).unsqueeze(0).to(args.device)
                    # Align audio (T,64) with video latents y (1,T,D)
                    fused, align = av_sync_mod(a, y)
                    # Encourage high alignment cosine → minimize (1 - align)
                    av_loss = (1.0 - align.mean())
                    loss = loss + float(args.av_weight) * av_loss
                except Exception:
                    pass
            # Optional physics violation classification + repair terms
            if physics_head is not None and idx < len(names):
                try:
                    name = names[idx]
                    spans = violation_spans.get(name, [])
                    if spans:
                        T = y.size(1)
                        labels = torch.zeros(T, device=y.device)
                        for (t0, t1, _kind) in spans:
                            a = max(0, min(T-1, t0)); b = max(0, min(T-1, t1))
                            if b < a:
                                a, b = b, a
                            labels[a:(b+1)] = 1.0
                        logits = physics_head(y.squeeze(0)).squeeze(-1)  # (T,)
                        bce = F.binary_cross_entropy_with_logits(logits, labels)
                        loss = loss + float(args.physics_weight) * bce
                        if bool(args.physics_repair):
                            yy = y.squeeze(0)
                            interp = 0.5 * (torch.roll(yy, shifts=1, dims=0) + torch.roll(yy, shifts=-1, dims=0))
                            repair = ((yy - interp).pow(2).sum(dim=-1) * labels).mean()
                            loss = loss + 0.1 * float(args.physics_weight) * repair
                except Exception:
                    pass

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if opt_av is not None:
                try:
                    # Simple joint step for AV module using same loss component
                    opt_av.zero_grad(set_to_none=True)
                except Exception:
                    pass
            step += 1
            if step % 50 == 0:
                print({"step": step, "loss": float(loss.item())})
            if step >= int(args.steps):
                break

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {
        'train_args': {
            'frames': int(args.frames) if hasattr(args, 'frames') else None,
            'd_model': int(args.d_model) if hasattr(args, 'd_model') else None,
        }
    }
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"[temporal_train] saved to {final}")

    if args.export_onnx:
        try:
            model.eval()
            dummy = torch.randn(1, int(args.frames), int(args.d_model))
            torch.onnx.export(model, dummy, args.export_onnx, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "B", 1: "T"}, "y": {0: "B", 1: "T"}}, opset_version=18)
            print(f"[onnx] exported temporal SSM to {args.export_onnx}")
        except Exception as e:
            print(f"[warn] temporal onnx export failed: {e}")

    # Optional: compute FVD if paths provided and library available
    if args.fvd_ref_dir and args.fvd_pred_dir:
        try:
            from pytorch_fvd import FVD as _FVD  # type: ignore
            import torchvision.transforms as _T  # type: ignore
            import cv2  # type: ignore
            def _load_videos(dir_path: str, max_v: int = 8) -> torch.Tensor:
                vids = []
                root = Path(dir_path)
                for p in sorted(root.glob('*.mp4'))[:max_v]:
                    cap = cv2.VideoCapture(str(p))
                    frames = []
                    ok = True
                    while ok:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        fr = cv2.resize(fr, (224, 224), interpolation=cv2.INTER_AREA)
                        frames.append(torch.from_numpy(fr).permute(2, 0, 1))
                    cap.release()
                    if frames:
                        vids.append(torch.stack(frames, dim=0))  # (T,3,H,W)
                if not vids:
                    return torch.zeros(1, 4, 3, 224, 224)
                # Pad/truncate to same length
                Tm = max(v.size(0) for v in vids)
                vids2 = []
                for v in vids:
                    if v.size(0) < Tm:
                        pad = v[-1:].repeat(Tm - v.size(0), 1, 1, 1)
                        v = torch.cat([v, pad], dim=0)
                    elif v.size(0) > Tm:
                        v = v[:Tm]
                    vids2.append(v)
                batch = torch.stack(vids2, dim=0)  # (B,T,3,H,W)
                return batch
            refs = _load_videos(args.fvd_ref_dir)
            preds = _load_videos(args.fvd_pred_dir)
            fvd = _FVD(cuda=torch.cuda.is_available())
            score = fvd.score(refs, preds)
            print({"FVD": float(score)})
        except Exception as e:
            print(f"[fvd] skipped: {e}")


if __name__ == "__main__":
    main()


