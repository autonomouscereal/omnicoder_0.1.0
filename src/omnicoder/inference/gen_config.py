from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GenRuntimeConfig:
    draft_ckpt_path: str | None = None
    draft_preset_name: str = 'mobile_2gb'
    use_onnx_draft: bool = False
    onnx_decode_path: str | None = None
    ort_provider: str = 'auto'
    super_verbose: bool | None = None
    # Hot-path knobs (avoid env reads)
    disable_disk_cache: bool = True
    graphrag_enable: bool = True
    rep_penalty: float = 1.1
    rep_window: int = 128
    no_repeat_ngram: int = 0
    no_repeat_window: int = 256
    min_p: float = 0.0
    freq_penalty: float = 0.0
    presence_penalty: float = 0.0
    mask_non_text: bool = False
    # CIS cache and long-context controls
    cis_cache_enable: bool = False
    cis_eps: float = 0.01
    cis_cap: int = 256
    use_landmarks_mode: str = 'auto'
    trace_enable: bool = True
    rg_accept_margin: float = 0.0
    latent_bfs_width: int = 3
    reflect_entropy_min: float = 2.0


def build_runtime_config_from_env(env: dict[str, str] | None = None) -> GenRuntimeConfig:
    e = env or os.environ
    def _g(k: str, d: str = '') -> str:
        return e.get(k, d)
    def _b(k: str, d: str = '0') -> bool:
        return _g(k, d) == '1'
    return GenRuntimeConfig(
        draft_ckpt_path=(_g('OMNICODER_DRAFT_CKPT', '').strip() or None),
        draft_preset_name=(_g('OMNICODER_DRAFT_PRESET', 'mobile_2gb') or 'mobile_2gb'),
        use_onnx_draft=_b('OMNICODER_USE_ONNX', '0'),
        onnx_decode_path=(_g('OMNICODER_API_ONNX_DECODE', '').strip() or None),
        ort_provider=_g('OMNICODER_ORT_PROVIDER', 'auto'),
        super_verbose=(_b('OMNICODER_GEN_SUPER_VERBOSE','0') or _b('OMNICODER_DEBUG_MODEL_VERBOSE','0')),
        disable_disk_cache=_b('OMNICODER_DISABLE_DISK_CACHE','1'),
        graphrag_enable=_b('OMNICODER_GRAPHRAG_ENABLE','1'),
        rep_penalty=float(_g('OMNICODER_REP_PENALTY','1.1') or '1.1'),
        rep_window=int(_g('OMNICODER_REP_WINDOW','128') or '128'),
        no_repeat_ngram=int(_g('OMNICODER_NO_REPEAT_NGRAM','0') or '0'),
        no_repeat_window=int(_g('OMNICODER_NO_REPEAT_WINDOW','256') or '256'),
        min_p=float(_g('OMNICODER_MIN_P','0.0') or '0.0'),
        freq_penalty=float(_g('OMNICODER_FREQ_PENALTY','0.0') or '0.0'),
        presence_penalty=float(_g('OMNICODER_PRESENCE_PENALTY','0.0') or '0.0'),
        mask_non_text=_b('OMNICODER_MASK_NON_TEXT','0'),
        cis_cache_enable=_b('OMNICODER_CIS_CACHE','0'),
        cis_eps=float(_g('OMNICODER_CIS_EPS','0.01') or '0.01'),
        cis_cap=int(_g('OMNICODER_CIS_CAP','256') or '256'),
        use_landmarks_mode=_g('OMNICODER_USE_LANDMARKS','auto') or 'auto',
        trace_enable=_b('OMNICODER_TRACE_ENABLE','1'),
        rg_accept_margin=float(_g('OMNICODER_RG_ACCEPT_MARGIN','0.0') or '0.0'),
        latent_bfs_width=int(_g('OMNICODER_LATENT_BFS_WIDTH','3') or '3'),
        reflect_entropy_min=float(_g('OMNICODER_REFLECT_ENTROPY_MIN','2.0') or '2.0'),
    )
