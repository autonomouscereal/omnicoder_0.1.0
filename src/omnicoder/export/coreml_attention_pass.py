from __future__ import annotations

"""
Core ML attention + RoPE mapping pass (best-effort).

Today this pass annotates a Core ML model (MLModel) with user-defined
metadata keys that downstream tooling/runtimes can use to select native
Apple attention ops and configure RoPE parameters.

Future work: replace Q/K/V + RoPE blocks with a native attention MIL op
when coremltools exposes stable APIs for programmatic graph rewrites.
"""

from typing import Any


def try_map_attention_with_rope(
    mlmodel: Any,
    *,
    rope_base: float = 10000.0,
    rope_scale: float = 1.0,
    kind: str = "rope_mqa",
    kv_latent_dim: int | None = None,
    multi_query: bool | None = True,
    fused_rope: bool | None = True,
) -> Any:
    """Annotate mlmodel with attention/RoPE metadata; return the same model.

    Parameters
    - mlmodel: coremltools.models.MLModel
    - rope_base: base for RoPE frequency calculation
    - rope_scale: scaling factor for long-context interpolation
    - kind: a short tag describing the attention mapping (e.g., 'rope_mqa')
    """
    try:
        md = mlmodel.user_defined_metadata  # type: ignore[attr-defined]
        md["omnicoder_attention"] = str(kind)
        md["rope_base"] = str(float(rope_base))
        md["rope_scale"] = str(float(rope_scale))
        if kv_latent_dim is not None:
            md["kv_latent_dim"] = str(int(kv_latent_dim))
        if multi_query is not None:
            md["multi_query"] = "1" if multi_query else "0"
        if fused_rope is not None:
            md["fused_rope"] = "1" if fused_rope else "0"
        # Optionally retain any pre-existing kv_latent_dim tag
        try:
            md.setdefault("kv_latent_dim", md.get("kv_latent_dim", ""))
        except Exception:
            pass
    except Exception:
        # Fallback to spec metadata if user_defined_metadata is unavailable
        try:
            spec = mlmodel.get_spec()  # type: ignore[attr-defined]
            meta = spec.description.metadata.user_defined  # type: ignore[attr-defined]
            meta["omnicoder_attention"] = str(kind)
            meta["rope_base"] = str(float(rope_base))
            meta["rope_scale"] = str(float(rope_scale))
            if kv_latent_dim is not None:
                meta["kv_latent_dim"] = str(int(kv_latent_dim))
            if multi_query is not None:
                meta["multi_query"] = "1" if multi_query else "0"
            if fused_rope is not None:
                meta["fused_rope"] = "1" if fused_rope else "0"
        except Exception:
            pass
    return mlmodel


def try_replace_qkv_with_native_attention(mlmodel: Any, *, op: str = "native_attention") -> Any:
    """Best-effort: mark model as using native attention.

    True graph replacement depends on coremltools MIL APIs and is not
    guaranteed across versions. Here we annotate the model to indicate
    a native attention mapping should be preferred by downstream tools.
    """
    try:
        md = mlmodel.user_defined_metadata  # type: ignore[attr-defined]
        md["native_attention"] = "1"
        md["attention_op"] = str(op)
        # Keep the mapping tag consistent
        md.setdefault("omnicoder_attention", md.get("omnicoder_attention", "rope_mqa"))
    except Exception:
        try:
            spec = mlmodel.get_spec()  # type: ignore[attr-defined]
            meta = spec.description.metadata.user_defined  # type: ignore[attr-defined]
            meta["native_attention"] = "1"
            meta["attention_op"] = str(op)
            if "omnicoder_attention" not in meta:
                meta["omnicoder_attention"] = "rope_mqa"
        except Exception:
            pass
    return mlmodel



