from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os

from . import FactorScore


@dataclass
class AudioSolver:
    def solve(self, factor: Any) -> FactorScore:
        # Prefer ASR WER JSONL if provided; otherwise FAD directories
        aux: Dict[str, Any] = {"prefer_strings": [], "avoid_strings": []}
        z = 0.0
        asr_jsonl = os.getenv('SFB_ASR_JSONL', '').strip()
        if asr_jsonl:
            try:
                import jiwer  # type: ignore
                from omnicoder.eval.audio_eval import _read_jsonl  # type: ignore
                # Configurable ASR backend: whisper|whisperx|adapter
                backend = os.getenv('SFB_ASR_BACKEND', 'whisper').strip().lower()
                pairs = _read_jsonl(asr_jsonl)
                refs, hyps = [], []
                if backend in ('whisper', 'whisperx'):
                    try:
                        import whisper  # type: ignore
                        model_name = os.getenv('SFB_ASR_MODEL', 'base')
                        model = whisper.load_model(model_name)
                        for rec in pairs[:10]:
                            hyp = model.transcribe(rec["file"]).get('text', '')
                            refs.append(rec.get("ref", ""))
                            hyps.append(str(hyp))
                    except Exception:
                        # Fallback to adapter if whisper not available
                        from omnicoder.modeling.multimodal.asr import ASRAdapter
                        asr = ASRAdapter(os.getenv('SFB_ASR_MODEL', 'small'))
                        for rec in pairs[:10]:
                            refs.append(rec.get("ref", ""))
                            hyps.append(asr.transcribe(rec["file"]) or "")
                else:
                    from omnicoder.modeling.multimodal.asr import ASRAdapter
                    asr = ASRAdapter(os.getenv('SFB_ASR_MODEL', 'small'))
                    for rec in pairs[:10]:
                        refs.append(rec.get("ref", ""))
                        hyps.append(asr.transcribe(rec["file"]) or "")
                wer = jiwer.wer(refs, hyps)
                z = float(1.0 - max(0.0, min(1.0, wer)))
            except Exception:
                z = 0.0
        else:
            pred = os.getenv('SFB_FAD_PRED_DIR', '').strip()
            ref = os.getenv('SFB_FAD_REF_DIR', '').strip()
            if pred and ref:
                try:
                    from omnicoder.eval import reward_metrics as _rm  # type: ignore
                    fad = _rm.fad(pred, ref)
                    if fad is not None:
                        z = float(1.0 / (1.0 + max(0.0, fad)))
                except Exception:
                    z = 0.0
        # Add small NMN-style suggestion for ASR/diarization when description available
        try:
            meta = getattr(factor, 'meta', {})
            desc = str(meta.get('desc', '')).lower()
            if 'transcrib' in desc or 'asr' in desc or 'speech' in desc:
                # Encourage transcription scaffolding
                aux["prefer_strings"] += ["Transcript:", "Speaker 1:", "Speaker 2:"]
                if os.getenv('SFB_DIARIZATION_HINTS', '1') == '1':
                    aux["prefer_strings"] += ["[00:00]", "[00:10]", "[00:20]"]
        except Exception:
            pass
        aux["audio_z"] = z
        return FactorScore(name="audio", score=0.05 + 0.2 * z, aux=aux)


