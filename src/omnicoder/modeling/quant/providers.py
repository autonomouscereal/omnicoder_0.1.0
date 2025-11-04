from __future__ import annotations

"""Provider-backed quantization helpers.

This module centralizes optional integrations with provider libraries,
e.g., bitsandbytes 4-bit linears on CUDA, and exposes helpers used by
model conversion utilities.
"""

from typing import Optional

import torch


def is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def try_bnb_linear4bit(linear: torch.nn.Linear):
    """Return a bitsandbytes 4-bit Linear replacement if available and on CUDA.

    Fallback: return None if bnb is not importable or CUDA is unavailable.
    """
    if not is_cuda_available():
        return None
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception:
        return None
    # Map torch.nn.Linear → bnb.nn.Linear4bit with nf4 or fp4 quantization
    # Use nf4 by default for better accuracy
    return bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        quant_type="nf4",
        compute_dtype=torch.float16,
    ).to(next(linear.parameters()).device)


def replace_linears_with_bnb_4bit(module: torch.nn.Module) -> int:
    """Recursively replace nn.Linear with bnb nn.Linear4bit where possible.

    Returns number of replacements.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            bnb_layer = try_bnb_linear4bit(child)
            if bnb_layer is not None:
                # Load weights
                with torch.no_grad():
                    # bnb expects weight in (out_features, in_features)
                    bnb_layer.weight.data.copy_(child.weight.data.to(bnb_layer.weight.dtype))
                    if child.bias is not None and hasattr(bnb_layer, "bias"):
                        bnb_layer.bias.data.copy_(child.bias.data.to(bnb_layer.bias.dtype))
                setattr(module, name, bnb_layer)
                replaced += 1
            else:
                # keep original
                pass
        else:
            replaced += replace_linears_with_bnb_4bit(child)
    return replaced



