from __future__ import annotations

"""Named wrappers for Omnicoder 2026 compressed global attention.

The executable implementation lives in ``omnicoder2026.py`` for checkpoint
stability. This module gives the architecture a clear import surface for native
runtime kernels and future CSA/HCA replacements.
"""

from omnicoder.modeling.omnicoder2026 import SparseLatentAttention

CSAAttention = SparseLatentAttention
HCAAttention = SparseLatentAttention

__all__ = ["SparseLatentAttention", "CSAAttention", "HCAAttention"]
