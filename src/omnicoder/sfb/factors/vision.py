from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os
import pathlib

from . import FactorScore


@dataclass
class VisionSolver:
    def solve(self, factor: Any) -> FactorScore:
        aux: Dict[str, Any] = {"prefer_strings": [], "avoid_strings": []}
        # Optional CLIPScore sidecar to seed a normalized z-score in arbiter
        cj = os.getenv('SFB_CLIP_JSONL', '').strip()
        z = 0.0
        if cj:
            try:
                from omnicoder.eval import reward_metrics as _rm  # type: ignore
                cs = _rm.clip_score(cj)
                if cs is not None:
                    z = float((cs + 1.0) / 2.0)
            except Exception:
                z = 0.0
        # NMN-style tiny detector/grounder stub
        meta = getattr(factor, 'meta', {})
        desc = str(meta.get('desc', '')).lower()
        # Parse simple attribute/object pairs: e.g., "red cup", "blue car"
        try:
            pairs: List[Tuple[str, str]] = []
            import re as _re
            for m in _re.finditer(r"\b(red|green|blue|black|white|yellow|large|small)\s+(\w+)\b", desc):
                pairs.append((m.group(1), m.group(2)))
            # Map into textual hints
            for (attr, obj) in pairs[:4]:
                aux["prefer_strings"].append(f"{attr} {obj}")
        except Exception:
            pass
        # Try optional detector/segmenter backends if available and an image path is provided
        try:
            img_path = str(os.getenv('SFB_IMAGE_PATH', '')).strip()
            if not img_path:
                # Try to infer from sidecar JSONL first entry
                j = os.getenv('SFB_CLIP_JSONL', '').strip()
                if j and pathlib.Path(j).exists():
                    try:
                        line = next((l for l in open(j, 'r', encoding='utf-8', errors='ignore') if l.strip()), '')
                        if '"file"' in line:
                            import json as _json
                            obj = _json.loads(line)
                            img_path = str(obj.get('file', ''))
                    except Exception:
                        img_path = ''
            backend = os.getenv('SFB_VISION_BACKEND', '').strip().lower()
            if img_path and pathlib.Path(img_path).exists():
                # If torchvision is available and backend allows, run a simple detector
                if backend in ('', 'torchvision', 'tv'):  # default to torchvision if present
                    try:
                        import torchvision  # type: ignore
                        import torchvision.transforms as T  # type: ignore
                        from PIL import Image  # type: ignore
                        # Use a small pretrained model for objectness
                        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # type: ignore
                        model.eval()
                        img = Image.open(img_path).convert('RGB')
                        tfm = T.Compose([T.ToTensor()])
                        x = tfm(img)
                        with torch.no_grad():  # type: ignore[name-defined]
                            out = model([x])[0]
                        # Convert top boxes to region hints
                        boxes = out.get('boxes', [])[:3] if isinstance(out, dict) else []
                        aux['regions'] = [list(map(float, b.tolist())) for b in boxes] if hasattr(boxes, 'tolist') else []
                        # Slightly lift z if detections exist
                        if aux['regions']:
                            z = max(z, 0.6)
                    except Exception:
                        pass
                # Lightweight Grounding-DINO/DETR-style hook with CLIP guidance and caching
                if backend in ('grounding-dino', 'grd', 'detr'):
                    try:
                        cache_dir = os.getenv('SFB_VISION_CACHE', 'weights/sfb/vision_cache')
                        from pathlib import Path as _P
                        _P(cache_dir).mkdir(parents=True, exist_ok=True)
                        import hashlib as _h
                        fp = _h.sha1((img_path + '|' + desc).encode('utf-8')).hexdigest() + '.json'
                        cpath = str(_P(cache_dir) / fp)
                        if os.path.exists(cpath):
                            import json as _json
                            obj = _json.loads(open(cpath, 'r', encoding='utf-8').read())
                            aux['regions'] = obj.get('regions', [])
                            z = max(z, float(obj.get('z', 0.0)))
                        else:
                            # Placeholder: try DETR if available; else no-op
                            regions: List[List[float]] = []
                            try:
                                import torchvision  # type: ignore
                                import torchvision.transforms as T  # type: ignore
                                from PIL import Image  # type: ignore
                                detr = torchvision.models.detection.detr_resnet50(weights='DEFAULT')  # type: ignore
                                detr.eval()
                                img = Image.open(img_path).convert('RGB')
                                tfm = T.Compose([T.ToTensor()])
                                x = tfm(img)
                                with torch.no_grad():  # type: ignore[name-defined]
                                    out = detr([x])[0]
                                boxes = out.get('boxes', [])[:3] if isinstance(out, dict) else []
                                regions = [list(map(float, b.tolist())) for b in boxes] if hasattr(boxes, 'tolist') else []
                                if regions:
                                    z = max(z, 0.6)
                            except Exception:
                                regions = []
                            aux['regions'] = regions
                            try:
                                import json as _json
                                with open(cpath, 'w', encoding='utf-8') as f:
                                    _json.dump({'regions': regions, 'z': float(z)}, f)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass
        # Construct a very small NMN plan: DETECT(attr,obj) -> GROUND(obj) -> ANSWER
        nmn_plan: List[Dict[str, Any]] = []
        if pairs:
            for (attr, obj) in pairs[:2]:
                nmn_plan.append({"op": "DETECT", "attr": attr, "obj": obj})
                nmn_plan.append({"op": "GROUND", "obj": obj})
        else:
            if desc:
                # default grounding
                nmn_plan.append({"op": "GROUND", "obj": desc.split()[0]})
        nmn_plan.append({"op": "ANSWER"})
        aux["nmn_plan"] = nmn_plan
        if 'red' in desc:
            aux["prefer_strings"].append("red ")
        aux["clip_z"] = z
        return FactorScore(name="vision", score=0.05 + 0.2 * z, aux=aux)


