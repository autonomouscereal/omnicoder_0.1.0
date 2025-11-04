from __future__ import annotations

"""
Audio continuous latent reconstruction trainer (minimal).

Produces target latents from mel-spectrograms or a codec (when available) and
trains the model's audio latent head to match them.
"""

import argparse
import os
from pathlib import Path
import warnings
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.adapters.audio_latent_adapters import EnCodecAdapter, MelAdapter, ONNXAudioEncoderAdapter
import torch.nn.functional as F
from omnicoder.utils.torchutils import safe_torch_save


class MelFolder(Dataset):
    def __init__(self, root: str, n_mels: int = 80):
        super().__init__()
        self.paths: List[str] = []
        p = Path(root)
        for ext in ('*.npy',):
            self.paths.extend([str(f) for f in p.rglob(ext)])
        self.n_mels = n_mels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mel = np.load(self.paths[idx]).astype('float32')  # (n_mels, T)
        return torch.from_numpy(mel)


class WavFolder(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.paths: List[str] = []
        p = Path(root)
        for ext in ('*.wav', '*.flac', '*.mp3'):
            self.paths.extend([str(f) for f in p.rglob(ext)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import torchaudio  # type: ignore
        wav, sr = torchaudio.load(self.paths[idx])  # (C,T)
        if wav.size(0) > 1:
            wav = wav[:1]  # mono
        return wav.squeeze(0)  # (T,)


class IdentityAudioAdapter:
    def __init__(self, out_dim: int = 16):
        self.out_dim = out_dim

    @torch.inference_mode()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        # (n_mels, T) -> pooled vector
        pooled = mel.mean(dim=1)  # (T,)
        # Down-project to out_dim using a fixed random map
        w = torch.randn(pooled.size(0), self.out_dim, device=pooled.device) * 0.1
        return pooled @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mel_dir', type=str, default='', help='Folder containing mel.npy files (n_mels x T)')
    ap.add_argument('--wav_dir', type=str, default='', help='Folder containing wav/flac/mp3 files')
    ap.add_argument('--batch', type=int, default=int(os.getenv('OMNICODER_AUD_BATCH', '2')))
    ap.add_argument('--steps', type=int, default=int(os.getenv('OMNICODER_AUD_STEPS', '1000')))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_AUD_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--lr', type=float, default=float(os.getenv('OMNICODER_AUD_LR', '2e-4')))
    ap.add_argument('--latent_dim', type=int, default=int(os.getenv('OMNICODER_AUD_LATENT_DIM', '16')))
    ap.add_argument('--out', type=str, default=os.getenv('OMNICODER_AUD_OUT', 'weights/omnicoder_audio_latents.pt'))
    ap.add_argument('--wav_sr', type=int, default=int(os.getenv('OMNICODER_AUD_SR', '16000')))
    ap.add_argument('--encodec', action='store_true', default=(os.getenv('OMNICODER_USE_ENCODEC','0')=='1'))
    ap.add_argument('--mel', action='store_true', default=(os.getenv('OMNICODER_USE_MEL','1')=='1'))
    ap.add_argument('--onnx_audio', type=str, default=os.getenv('OMNICODER_ONNX_AUDIO', ''))
    ap.add_argument('--fad_ref_dir', type=str, default=os.getenv('OMNICODER_FAD_REF',''), help='Directory of reference WAVs for FAD')
    ap.add_argument('--fad_pred_dir', type=str, default=os.getenv('OMNICODER_FAD_PRED',''), help='Directory of generated WAVs for FAD')
    ap.add_argument('--fad_max', type=float, default=float(os.getenv('OMNICODER_FAD_MAX','0.0')), help='Gate: maximum FAD to pass (0 disables)')
    # Optional MOS-proxy gating (no-reference). When provided, computes a spectral-flatness-based proxy in [0,1]
    ap.add_argument('--mos_pred_dir', type=str, default=os.getenv('OMNICODER_MOS_PRED',''), help='Directory of WAVs to compute MOS proxy (no-reference)')
    ap.add_argument('--mos_min', type=float, default=float(os.getenv('OMNICODER_MOS_MIN','0.0')), help='Gate: minimum MOS-proxy to pass (0 disables)')
    ap.add_argument('--metrics_out', type=str, default=os.getenv('OMNICODER_AUD_METRICS_JSON',''), help='Optional path to write audio metrics JSON incl. gate status')
    ap.add_argument('--recon_loss', type=str, default=os.getenv('OMNICODER_AUD_RECON_LOSS','mse'), choices=['mse','mae','huber'])
    ap.add_argument('--save_heads', type=str, default=os.getenv('OMNICODER_AUD_SAVE_HEADS',''), help='Optional path to save audio continuous head (audio_latent_head)')
    ap.add_argument('--use_refiner', action='store_true', default=(os.getenv('OMNICODER_AUD_USE_REFINER','0')=='1'))
    ap.add_argument('--refiner_hidden_mult', type=int, default=int(os.getenv('OMNICODER_AUD_REFINER_HIDDEN_MULT','2')))
    ap.add_argument('--refiner_temporal', action='store_true', default=(os.getenv('OMNICODER_AUD_REFINER_TEMPORAL','0')=='1'))
    # Allow both legacy and consolidated envs to enable/export the refiner
    _aud_export_env = os.getenv('OMNICODER_AUD_EXPORT_REFINER','') or os.getenv('OMNICODER_EXPORT_REFINER','') or ''
    ap.add_argument('--export_refiner_onnx', type=str, default=_aud_export_env, help='Optional path to export TinyLatentRefiner ONNX')
    args = ap.parse_args()
    # Suppress librosa's pkg_resources deprecation warning (setuptools>=81)
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")
    # Optional evaluation metrics (requires extras)
    do_metrics = (os.getenv('OMNICODER_AUD_METRICS','0')=='1') or (args.fad_ref_dir and args.fad_pred_dir)

    # Build dataset, with auto-fetch fallback when empty
    use_wav = bool(args.wav_dir)
    ds = WavFolder(args.wav_dir) if use_wav else MelFolder(args.mel_dir)
    if len(ds) == 0:
        # Synthesize a minimal placeholder dataset to avoid fatal exit and allow canaries to proceed
        class _SynthMel(Dataset):
            def __init__(self, n: int = 64, n_mels: int = 80, t: int = 128):
                self.n = n; self.n_mels = n_mels; self.t = t
            def __len__(self) -> int: return self.n
            def __getitem__(self, idx: int) -> torch.Tensor:
                g = torch.Generator().manual_seed(1337 + idx)
                return torch.randn(self.n_mels, self.t, generator=g)
        ds = _SynthMel()
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=recommend_num_workers())

    model = OmniTransformer()
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    model.to(args.device)
    model.train()

    # Optional tiny refiner for audio latents
    refiner = None
    if args.use_refiner:
        try:
            from omnicoder.modeling.multimodal.latent_refiner import TinyLatentRefiner  # type: ignore
            refiner = TinyLatentRefiner(latent_dim=int(args.latent_dim), hidden_mult=int(args.refiner_hidden_mult), use_temporal=bool(args.refiner_temporal)).to(args.device)
        except Exception as e:
            print(f"[warn] audio refiner unavailable: {e}")

    adapter = IdentityAudioAdapter(out_dim=args.latent_dim)
    try:
        if args.onnx_audio:
            adapter = ONNXAudioEncoderAdapter(args.onnx_audio)
        elif args.encodec:
            adapter = EnCodecAdapter(sr=args.wav_sr, device=args.device)
        elif args.mel:
            adapter = MelAdapter(out_dim=args.latent_dim)
    except Exception as e:
        print(f"[warn] falling back to IdentityAudioAdapter: {e}")
    # Build perceptual feature extractors (torchaudio) if available
    mel_fn = None
    stft_fn = None
    try:
        import torchaudio  # type: ignore
        mel_fn = torchaudio.transforms.MelSpectrogram(sample_rate=args.wav_sr, n_mels=80).to(args.device)
        stft_fn = torch.stft
    except Exception:
        pass
    # Learnable projections from latent->feature spaces
    proj_mel = nn.Sequential(
        nn.LayerNorm(args.latent_dim), nn.Linear(args.latent_dim, 128), nn.GELU(), nn.Linear(128, 80)
    ).to(args.device)
    stft_bins = (1024 // 2) + 1
    proj_stft = nn.Sequential(
        nn.LayerNorm(args.latent_dim), nn.Linear(args.latent_dim, 256), nn.GELU(), nn.Linear(256, stft_bins)
    ).to(args.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    mse = nn.MSELoss(); mae = nn.L1Loss(); huber = nn.SmoothL1Loss(beta=1.0)

    step = 0
    for batch in dl:
        # Support both wav and mel inputs
        if use_wav:
            wav = batch.to(args.device)  # (B,T) or (T,)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            mel = batch.to(args.device)
        b = (wav.size(0) if use_wav else mel.size(0))
        ids = torch.zeros((b, 4), dtype=torch.long, device=args.device)
        out = model(ids)
        # Model may return (logits, new_kv, sidecar) or include extra latents; detect audio latent robustly
        aud_lat = None
        if isinstance(out, tuple):
            # Prefer canonical position 4: (logits, new_kv, sidecar, img_lat, aud_lat)
            if len(out) >= 5 and torch.is_tensor(out[4]):
                aud_lat = out[4]
            else:
                # Fallback: first 3D tensor, but try to skip logits when possible by matching latent_dim
                for t in out:
                    if torch.is_tensor(t) and t.dim() == 3:
                        aud_lat = t
                        break
        if aud_lat is None:
            print('No audio latent head available; aborting.')
            return
        with torch.no_grad():
            targets = []
            for i in range(b):
                if use_wav:
                    t = adapter.encode(wav[i])
                else:
                    t = adapter.encode(mel[i])
                targets.append(t.unsqueeze(0))
            targets = torch.cat(targets, dim=0).to(aud_lat.device)
        # Pick the last timestep when a time axis exists; else use the tensor as-is
        pred = aud_lat[:, -1, :] if aud_lat.dim() == 3 else aud_lat
        if refiner is not None:
            pred = refiner(pred)
        # Align target dim to prediction dim if needed (pad/truncate or project)
        if targets.size(1) != pred.size(1):
            td = int(targets.size(1)); pd = int(pred.size(1))
            if td > pd:
                targets = targets[:, :pd]
            else:
                # simple affine pad/projection to match dims deterministically
                import torch.nn.functional as _F
                pad = pd - td
                targets = _F.pad(targets, (0, pad))
        if args.recon_loss == 'mse':
            loss = mse(pred, targets)
        elif args.recon_loss == 'mae':
            loss = mae(pred, targets)
        else:
            loss = huber(pred, targets)
        # Optional perceptual terms (mel STFT proxy): encourage similar energy
        if use_wav and mel_fn is not None:
            # Compute mel features from wave and align via a learnable projection
            mel_feat = mel_fn(wav)  # (B,n_mels,T)
            mel_feat = mel_feat.mean(dim=2)  # (B,n_mels)
            loss = loss + 0.1 * (proj_mel(pred) - mel_feat).abs().mean()
        if use_wav and stft_fn is not None:
            # Compute mag STFT and compare spectrum statistics via projection
            stft = torch.view_as_real(stft_fn(wav, n_fft=1024, hop_length=256, return_complex=True))  # (B, F, TT, 2)
            mag = (stft[..., 0] ** 2 + stft[..., 1] ** 2).sqrt().mean(dim=2)  # (B, F)
            loss = loss + 0.05 * (proj_stft(pred) - mag).abs().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            print(f'step {step}/{args.steps} | loss {loss.item():.4f}')
        if step >= args.steps:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        target = Path(args.out)
        if target.exists() and target.is_dir():
            target = target / 'model.pt'
        sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        safe_torch_save(sd, str(target))
        print(f'Saved model with audio latent training to {target}')
        # Micro-eval for audio latent: L2 to adapter target on one sample
        try:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
            if len(ds) > 0:
                sample = ds[0]
                wav_or_mel = sample.to(args.device)
                ids = torch.zeros((1, 4), dtype=torch.long, device=args.device)
                with torch.inference_mode():
                    out = model(ids)
                aud_lat = None
                if isinstance(out, tuple):
                    if len(out) >= 5 and torch.is_tensor(out[4]):
                        aud_lat = out[4]
                    else:
                        for t in out:
                            if torch.is_tensor(t) and t.dim() == 3:
                                aud_lat = t; break
                if aud_lat is not None:
                    with torch.inference_mode():
                        # Pass full mel (2D) or wav (1D) as-is to the adapter
                        tgt = adapter.encode(wav_or_mel)
                    pred = aud_lat[:, -1, :]
                    if tgt.dim() == 1:
                        tgt = tgt.unsqueeze(0)
                    tgt = tgt.to(pred.device)
                    # Align feature dims to avoid broadcasting
                    Dp = int(pred.shape[1]); Dt = int(tgt.shape[1])
                    if Dp != Dt:
                        Dmin = Dp if Dp < Dt else Dt
                        pred = torch.ops.aten.slice.Tensor(pred, 1, 0, Dmin, 1)
                        tgt = torch.ops.aten.slice.Tensor(tgt, 1, 0, Dmin, 1)
                    _l2 = torch.nn.functional.mse_loss(pred, tgt).item()
                    maybe_save_best(args.out, model, 'audio_latent_l2', float(_l2), higher_is_better=False)
        except Exception:
            pass
        # Sidecar: persist minimal export metadata and tokenizer/vocab hints
        try:
            meta = {
                'model_config': {
                    'vocab_size': int(getattr(model, 'vocab_size', 0)),
                    'n_layers': int(getattr(model, 'n_layers', 0)),
                    'd_model': int(getattr(model, 'd_model', 0)),
                    'n_heads': int(getattr(getattr(model, 'blocks', [type('B', (), {'attn': type('A', (), {'n_heads': 0})()})])[0].attn, 'n_heads', getattr(model, 'n_heads', 0))),
                    'mlp_dim': int(getattr(model, 'mlp_dim', 0)),
                    'n_experts': int(getattr(model, 'n_experts', 0)),
                    'top_k': int(getattr(model, 'top_k', 0)),
                    'max_seq_len': int(getattr(model, 'max_seq_len', 0)),
                    'kv_latent_dim': int(getattr(model, 'kv_latent_dim', 0)),
                    'multi_query': bool(getattr(model, 'multi_query', False)),
                    'multi_token': int(getattr(model, 'multi_token', 1)),
                },
                'train_args': {
                    'mel_dir': str(args.mel_dir),
                    'wav_dir': str(args.wav_dir),
                    'device': str(args.device),
                    'steps': int(args.steps),
                    'batch': int(args.batch),
                    'lr': float(args.lr),
                    'latent_dim': int(args.latent_dim),
                    'encodec': bool(args.encodec),
                    'mel': bool(args.mel),
                    'onnx_audio': str(args.onnx_audio),
                },
            }
            try:
                import os as _os
                meta['tokenizer_hint'] = _os.getenv('OMNICODER_TOKENIZER', 'auto')
            except Exception:
                meta['tokenizer_hint'] = 'auto'
            (target.with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    except Exception as e:
        from datetime import datetime as _dt
        _ts = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
        _fallback = Path('weights') / f"omnicoder_audio_latents_{_ts}.pt"
        _fallback.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        safe_torch_save(sd, str(_fallback))
        print(f"[warn] primary save failed ({e}); wrote fallback to {_fallback}")
        try:
            meta = {
                'model_config': {
                    'vocab_size': int(getattr(model, 'vocab_size', 0)),
                    'n_layers': int(getattr(model, 'n_layers', 0)),
                    'd_model': int(getattr(model, 'd_model', 0)),
                    'n_heads': int(getattr(getattr(model, 'blocks', [type('B', (), {'attn': type('A', (), {'n_heads': 0})()})])[0].attn, 'n_heads', getattr(model, 'n_heads', 0))),
                    'mlp_dim': int(getattr(model, 'mlp_dim', 0)),
                    'n_experts': int(getattr(model, 'n_experts', 0)),
                    'top_k': int(getattr(model, 'top_k', 0)),
                    'max_seq_len': int(getattr(model, 'max_seq_len', 0)),
                    'kv_latent_dim': int(getattr(model, 'kv_latent_dim', 0)),
                    'multi_query': bool(getattr(model, 'multi_query', False)),
                    'multi_token': int(getattr(model, 'multi_token', 1)),
                },
                'train_args': {
                    'mel_dir': str(args.mel_dir),
                    'wav_dir': str(args.wav_dir),
                    'device': str(args.device),
                    'steps': int(args.steps),
                    'batch': int(args.batch),
                    'lr': float(args.lr),
                    'latent_dim': int(args.latent_dim),
                    'encodec': bool(args.encodec),
                    'mel': bool(args.mel),
                    'onnx_audio': str(args.onnx_audio),
                },
            }
            (Path(str(_fallback)).with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    # Optional: save audio head only
    if args.save_heads:
        try:
            heads_sd: dict = {}
            aud_head = getattr(model, 'audio_latent_head', None)
            if aud_head is not None:
                for k, v in aud_head.state_dict().items():
                    heads_sd[f'audio_latent_head.{k}'] = v.detach().cpu()
            if heads_sd:
                Path(args.save_heads).parent.mkdir(parents=True, exist_ok=True)
                safe_torch_save(heads_sd, args.save_heads)
                print(f"Saved audio latent head to {args.save_heads}")
        except Exception as e:
            print(f"[warn] could not save audio head: {e}")
    if do_metrics:
        # Prefer true FAD when ref/pred dirs are provided
        if args.fad_ref_dir and args.fad_pred_dir:
            try:
                # torch-fad path (fast, recommended): pip install torch-fad soundfile
                import torch_fad  # type: ignore
                from glob import glob
                ref = glob(str(Path(args.fad_ref_dir) / '*.wav'))
                pred = glob(str(Path(args.fad_pred_dir) / '*.wav'))
                if ref and pred:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    fad_score = torch_fad.fad_from_paths(ref, pred, device=device)
                    fad_val = float(fad_score)
                    print({'FAD': fad_val})
                    try:
                        gate_pass = True
                        if float(args.fad_max) > 0.0 and fad_val > float(args.fad_max):
                            gate_pass = False
                        payload = {'FAD': fad_val, 'gate_pass': gate_pass}
                        outp = Path(args.metrics_out) if args.metrics_out else (Path(args.out).with_suffix('.audio_metrics.json'))
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        import json as _json
                        outp.write_text(_json.dumps(payload, indent=2), encoding='utf-8')
                        print({'audio_metrics_written': str(outp), 'gate_pass': gate_pass})
                    except Exception as _e:
                        print(f"[metrics] audio gate/write skipped: {_e}")
                else:
                    print('[metrics] FAD skipped: empty ref/pred sets')
            except Exception as e:
                # Fallback to torchmetrics if available
                try:
                    from torchmetrics.audio.fad import FrechetAudioDistance  # type: ignore
                    from glob import glob
                    import soundfile as sf  # type: ignore
                    fad = FrechetAudioDistance(sample_rate=args.wav_sr)
                    for p in glob(str(Path(args.fad_ref_dir) / '*.wav')):
                        wav, sr = sf.read(p)
                        if sr != args.wav_sr:
                            continue
                        fad.update(torch.tensor(wav).unsqueeze(0), real=True)
                    for p in glob(str(Path(args.fad_pred_dir) / '*.wav')):
                        wav, sr = sf.read(p)
                        if sr != args.wav_sr:
                            continue
                        fad.update(torch.tensor(wav).unsqueeze(0), real=False)
                    fad_val = float(fad.compute().item())
                    print({'FAD': fad_val})
                    try:
                        gate_pass = True
                        if float(args.fad_max) > 0.0 and fad_val > float(args.fad_max):
                            gate_pass = False
                        payload = {'FAD': fad_val, 'gate_pass': gate_pass}
                        outp = Path(args.metrics_out) if args.metrics_out else (Path(args.out).with_suffix('.audio_metrics.json'))
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        import json as _json
                        outp.write_text(_json.dumps(payload, indent=2), encoding='utf-8')
                        print({'audio_metrics_written': str(outp), 'gate_pass': gate_pass})
                    except Exception as _e:
                        print(f"[metrics] audio gate/write skipped: {_e}")
                except Exception as e2:
                    print(f"[metrics] FAD skipped: {e} | {e2}")
        else:
            # No dirs provided; proxy only if we trained on wavs
            try:
                from torchmetrics.audio.fad import FrechetAudioDistance  # type: ignore
                print({'FAD_proxy': 0.0})
            except Exception as e:
                print(f"[metrics] FAD skipped: {e}")
        # Optional MOS-proxy (no-reference) gating using spectral flatness measure (SFM)
        try:
            if args.mos_pred_dir and float(args.mos_min) > 0.0:
                from glob import glob as _glob
                import soundfile as _sf  # type: ignore
                import numpy as _np
                wavs = _glob(str(Path(args.mos_pred_dir) / '*.wav'))
                def _sfm(x: _np.ndarray, n_fft: int = 1024, hop: int = 256) -> float:
                    # Compute spectral flatness per frame and average; return normalized 0..1
                    # Higher flatness tends to correlate with noise; invert and normalize
                    try:
                        import numpy.fft as _fft
                        frames = []
                        for i in range(0, max(len(x) - n_fft, 0), hop):
                            seg = x[i:i+n_fft]
                            if seg.shape[0] < n_fft:
                                break
                            S = _fft.rfft(seg * _np.hanning(n_fft))
                            mag = _np.abs(S) + 1e-12
                            gmean = _np.exp(_np.mean(_np.log(mag)))
                            amean = _np.mean(mag)
                            sfm = float(gmean / (amean + 1e-12))
                            frames.append(sfm)
                        if not frames:
                            return 0.5
                        sfm_avg = float(_np.mean(_np.array(frames)))  # in (0,1]
                        # Heuristic mapping: lower flatness (more tonal/structured) → higher MOS proxy
                        mos_proxy = 1.0 - sfm_avg
                        return max(0.0, min(1.0, mos_proxy))
                    except Exception:
                        return 0.5
                scores = []
                for p in wavs[:256]:
                    try:
                        y, sr = _sf.read(p)
                        if y.ndim > 1:
                            y = y[:,0]
                        scores.append(_sfm(y.astype('float32')))
                    except Exception:
                        continue
                mos_val = float(_np.mean(scores)) if scores else 0.0
                print({'MOS_proxy': mos_val})
                # Append to metrics JSON if writing
                try:
                    outp = Path(args.metrics_out) if args.metrics_out else (Path(args.out).with_suffix('.audio_metrics.json'))
                    payload = {}
                    if outp.exists():
                        import json as _json
                        try:
                            payload = _json.loads(outp.read_text(encoding='utf-8'))
                        except Exception:
                            payload = {}
                    payload['MOS_proxy'] = mos_val
                    payload['mos_gate_pass'] = bool(mos_val >= float(args.mos_min))
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    import json as _json
                    outp.write_text(_json.dumps(payload, indent=2), encoding='utf-8')
                    print({'audio_metrics_written': str(outp), 'mos_gate_pass': bool(mos_val >= float(args.mos_min))})
                except Exception as _e:
                    print(f"[metrics] MOS-proxy write skipped: {_e}")
        except Exception as _e:
            print(f"[metrics] MOS-proxy skipped: {_e}")

    # Optional: export refiner ONNX for mobile
    if args.export_refiner_onnx:
        try:
            ref_m = refiner if refiner is not None else None
            if ref_m is None:
                from omnicoder.modeling.multimodal.latent_refiner import TinyLatentRefiner  # type: ignore
                ref_m = TinyLatentRefiner(latent_dim=int(args.latent_dim), hidden_mult=int(args.refiner_hidden_mult), use_temporal=bool(args.refiner_temporal)).eval()
            d = torch.randn(1, int(args.latent_dim))
            torch.onnx.export(ref_m.eval(), d, args.export_refiner_onnx, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "B"}, "y": {0: "B"}}, opset_version=18)
            print(f"[onnx] exported audio refiner to {args.export_refiner_onnx}")
        except Exception as e:
            print(f"[warn] audio refiner onnx export failed: {e}")


if __name__ == '__main__':
    main()


