from __future__ import annotations

"""
Reward modules for RL training and evaluation:
- CLIPScore for image-text alignment
- FID for image distributions
- FVD (Fréchet Video Distance) for video
- FAD (Fréchet Audio Distance) for audio

These functions are best-effort wrappers that use the standard libraries when installed.
They return None if a metric cannot be computed due to missing dependencies.
"""

from pathlib import Path
from typing import List, Optional


def clip_score(jsonl_pairs: str) -> Optional[float]:
    try:
        import json
        import torch
        from PIL import Image  # type: ignore
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scores: List[float] = []
        # Try OpenAI CLIP, fallback to open-clip-torch
        try:
            import clip  # type: ignore
            model, preprocess = clip.load('ViT-B/32', device=device)
            tokenize = clip.tokenize  # type: ignore
            use_openclip = False
        except Exception:
            import open_clip  # type: ignore
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai', device=device)  # type: ignore
            tokenizer = open_clip.get_tokenizer('ViT-B-32')  # type: ignore
            def tokenize(texts: List[str]):
                return tokenizer(texts)
            use_openclip = True
        model.eval()
        for line in open(jsonl_pairs, 'r', encoding='utf-8', errors='ignore'):
            if not line.strip():
                continue
            rec = json.loads(line)
            img = preprocess(Image.open(rec['file']).convert('RGB')).unsqueeze(0).to(device)
            text = tokenize([rec['prompt']]).to(device)
            with torch.no_grad():
                if use_openclip:
                    img_f = model.encode_image(img)
                    txt_f = model.encode_text(text)
                else:
                    img_f = model.encode_image(img)
                    txt_f = model.encode_text(text)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                sim = (img_f @ txt_f.T).squeeze().item()
                scores.append(sim)
        return float(sum(scores) / max(1, len(scores))) if scores else None
    except Exception:
        return None


def fid(pred_dir: str, ref_dir: str) -> Optional[float]:
    try:
        from cleanfid import fid as cleanfid  # type: ignore
        score = cleanfid.compute_fid(pred_dir, ref_dir, mode='clean')
        return float(score)
    except Exception:
        return None


def fvd(pred_dir: str, ref_dir: str) -> Optional[float]:
    try:
        import torch
        import pytorch_fvd  # type: ignore
        import numpy as np
        from glob import glob
        from PIL import Image  # type: ignore
        # Load videos as 16-frame clips at 224x224
        def _load_frames(path: str) -> torch.Tensor:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(path)
            frames = []
            i = 0
            while i < 16:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                fr = cv2.resize(fr, (224, 224))
                frames.append(fr)
                i += 1
            cap.release()
            if not frames:
                frames = [np.zeros((224,224,3), dtype=np.uint8) for _ in range(16)]
            arr = np.stack(frames, axis=0).astype('float32') / 255.0
            return torch.from_numpy(arr).permute(0,3,1,2)  # (T,3,H,W)
        pred_v = [p for p in Path(pred_dir).glob('*.mp4')]
        ref_v = [p for p in Path(ref_dir).glob('*.mp4')]
        if not pred_v or not ref_v:
            return None
        pred_t = [ _load_frames(str(p)) for p in pred_v ]
        ref_t = [ _load_frames(str(p)) for p in ref_v ]
        score = pytorch_fvd.fvd(pred_t, ref_t, device='cuda' if torch.cuda.is_available() else 'cpu')
        return float(score.item())
    except Exception:
        return None


def fad(pred_dir: str, ref_dir: str) -> Optional[float]:
    try:
        # FAD reference implementations depend on VGGish; here call torch-fad if available
        import torch
        import numpy as np
        from glob import glob
        import soundfile as sf  # type: ignore
        import torch_fad  # type: ignore
        pred = glob(str(Path(pred_dir) / '*.wav'))
        ref = glob(str(Path(ref_dir) / '*.wav'))
        if not pred or not ref:
            return None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        score = torch_fad.fad_from_paths(ref, pred, device=device)
        return float(score)
    except Exception:
        return None


