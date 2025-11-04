from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class ProofMarginInputs:
    llm_confidence: float
    sum_log_factors: float
    verifier_score: float
    retrieval_hits: int
    code_passk: float = 0.0
    clip_z: float = 0.0
    audio_z: float = 0.0
    video_z: float = 0.0


@dataclass
class ProofMarginArbiter:
    w_text: float = 1.0
    w_code: float = 1.0
    w_img: float = 1.0
    w_audio: float = 1.0
    w_video: float = 1.0
    threshold: Optional[float] = None
    learned_w_text: float | None = None
    learned_w_code: float | None = None
    learned_w_img: float | None = None
    learned_w_audio: float | None = None
    learned_w_video: float | None = None

    def __post_init__(self) -> None:
        # Use auto thresholding by default; allow override via env
        env_thresh = os.getenv("SFB_PROOF_MARGIN", "auto")
        if env_thresh != "auto":
            try:
                self.threshold = float(env_thresh)
            except Exception:
                self.threshold = None
        # Optional: load learned weights from JSON sidecar
        try:
            jpath = os.getenv('SFB_ARBITER_WEIGHTS', '').strip()
            if jpath:
                import json as _json
                cfg = _json.loads(open(jpath, 'r', encoding='utf-8').read())
                def _getf(k: str) -> Optional[float]:
                    try:
                        return float(cfg.get(k)) if k in cfg else None
                    except Exception:
                        return None
                self.learned_w_text = _getf('w_text')
                self.learned_w_code = _getf('w_code')
                self.learned_w_img = _getf('w_img')
                self.learned_w_audio = _getf('w_audio')
                self.learned_w_video = _getf('w_video')
        except Exception:
            pass

    def compute_margin(self, x: ProofMarginInputs) -> float:
        # Combine LLM confidence, verifier score, factor scores, and optional modality signals
        wt = float(self.learned_w_text if self.learned_w_text is not None else self.w_text)
        wc = float(self.learned_w_code if self.learned_w_code is not None else self.w_code)
        wi = float(self.learned_w_img if self.learned_w_img is not None else self.w_img)
        wa = float(self.learned_w_audio if self.learned_w_audio is not None else self.w_audio)
        wv = float(self.learned_w_video if self.learned_w_video is not None else self.w_video)
        return (
            wt * (x.llm_confidence + x.verifier_score)
            + x.sum_log_factors
            + wc * x.code_passk
            + wi * x.clip_z
            + wa * x.audio_z
            + wv * x.video_z
        )

    def accept(self, x: ProofMarginInputs) -> bool:
        m = self.compute_margin(x)
        if self.threshold is None:
            # Auto: accept if margin improved vs a small baseline
            return m >= 0.0
        return m >= float(self.threshold)

    def suggestions(self, x: ProofMarginInputs) -> Dict[str, Any]:
        """Return backpressure/control suggestions when margin is low.

        Suggestions: {escalate_block_verify, disable_early_exit, bump_scmoe_topk, request_resolve, request_retrieval}
        """
        out: Dict[str, Any] = {}
        try:
            m = self.compute_margin(x)
            thr = float(self.threshold) if self.threshold is not None else 0.0
            low = (m < thr)
            # Heuristics
            if low:
                out['escalate_block_verify'] = True
                out['disable_early_exit'] = True
                if x.code_passk < 0.5 and x.llm_confidence < 0.7:
                    out['bump_scmoe_topk'] = True
                if x.sum_log_factors < 0.0:
                    out['request_resolve'] = True
                if x.llm_confidence < 0.5:
                    out['request_retrieval'] = True
        except Exception:
            return {}
        return out

    @staticmethod
    def static_metrics_from_env() -> Dict[str, float]:
        """Optionally compute static quality metrics from environment hints.
        Returns zeros if inputs are missing. Intended as priors for gating.
        """
        out: Dict[str, float] = {
            'code_passk': 0.0,
            'clip_z': 0.0,
            'audio_z': 0.0,
            'video_z': 0.0,
        }
        try:
            from omnicoder.eval import reward_metrics as _rm  # type: ignore
        except Exception:
            return out
        import os as _os
        # CLIPScore prior (already ~[-1,1]); map to [0,1]
        cj = _os.getenv('SFB_CLIP_JSONL', '').strip()
        if cj:
            try:
                cs = _rm.clip_score(cj)
                if cs is not None:
                    out['clip_z'] = float((cs + 1.0) / 2.0)
            except Exception:
                pass
        # FVD: lower is better; map to [0,1] by inverse with soft normalization
        fvd_pred = _os.getenv('SFB_FVD_PRED_DIR', '').strip()
        fvd_ref = _os.getenv('SFB_FVD_REF_DIR', '').strip()
        if fvd_pred and fvd_ref:
            try:
                vd = _rm.fvd(fvd_pred, fvd_ref)
                if vd is not None:
                    out['video_z'] = float(1.0 / (1.0 + max(0.0, vd)))
            except Exception:
                pass
        # FAD: lower is better
        fad_pred = _os.getenv('SFB_FAD_PRED_DIR', '').strip()
        fad_ref = _os.getenv('SFB_FAD_REF_DIR', '').strip()
        if fad_pred and fad_ref:
            try:
                ad = _rm.fad(fad_pred, fad_ref)
                if ad is not None:
                    out['audio_z'] = float(1.0 / (1.0 + max(0.0, ad)))
            except Exception:
                pass
        # Code pass@k from a JSONL with {candidates:[], tests:"..."} rows
        code_tasks = _os.getenv('SFB_CODE_TASKS_JSONL', '').strip()
        if code_tasks:
            try:
                import json as _json
                from omnicoder.eval.code_eval import pass_at_k as _pass_at_k  # type: ignore
                rows = [_json.loads(l) for l in open(code_tasks, 'r', encoding='utf-8', errors='ignore') if l.strip()]
                ok = 0
                for ex in rows[:10]:  # clip to 10 to keep startup light
                    if _pass_at_k(ex.get('candidates', []), ex.get('tests', ''), k=5, timeout=3):
                        ok += 1
                if rows:
                    out['code_passk'] = float(ok / len(rows))
            except Exception:
                pass
        return out

    def log_decision(self, x: ProofMarginInputs, accepted: bool) -> None:
        """Telemetry: write a compact JSONL record when enabled.
        Controlled by SFB_ARBITER_LOG=/path (file) or empty to disable.
        """
        try:
            path = os.getenv('SFB_ARBITER_LOG', '').strip()
            if not path:
                return
            rec = {
                'llm_conf': float(x.llm_confidence),
                'sum_log_phi': float(x.sum_log_factors),
                'verifier': float(x.verifier_score),
                'retrieval_hits': int(x.retrieval_hits),
                'code': float(x.code_passk),
                'clip': float(x.clip_z),
                'audio': float(x.audio_z),
                'video': float(x.video_z),
                'accepted': bool(accepted),
            }
            with open(path, 'a', encoding='utf-8') as f:
                f.write(str(rec).replace("'", '"') + "\n")
        except Exception:
            pass

    @staticmethod
    def fit_weights_from_log(log_path: str) -> Dict[str, float]:
        """Best-effort fit of linear weights to maximize separation between accepted and rejected.

        Expects log records written by log_decision. Returns a dict with learned weights.
        """
        rows: list[dict[str, Any]] = []
        try:
            for line in open(log_path, 'r', encoding='utf-8', errors='ignore'):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            return {}
        if not rows:
            return {}
        # Features: [llm_conf+verifier, sum_log_phi, code, clip, audio, video]
        w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        lr = 0.05
        def dot(ws, xs):
            return sum(float(a) * float(b) for a, b in zip(ws, xs))
        def sigmoid(z):
            if z >= 0:
                e = 2.718281828 ** (-z)
                return 1.0 / (1.0 + e)
            e = 2.718281828 ** (z)
            return e / (1.0 + e)
        for _ in range(20):
            for r in rows:
                x = [
                    float(r.get('llm_conf', 0.0)) + float(r.get('verifier', 0.0)),
                    float(r.get('sum_log_phi', 0.0)),
                    float(r.get('code', 0.0)),
                    float(r.get('clip', 0.0)),
                    float(r.get('audio', 0.0)),
                    float(r.get('video', 0.0)),
                ]
                y = 1.0 if bool(r.get('accepted')) else 0.0
                p = sigmoid(dot(w, x))
                g = [(y - p) * v * lr for v in x]
                w = [a + b for (a, b) in zip(w, g)]
        return {
            'w_text': float(w[0]),
            'w_code': float(w[2]),
            'w_img': float(w[3]),
            'w_audio': float(w[4]),
            'w_video': float(w[5]),
        }


