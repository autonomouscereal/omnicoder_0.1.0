from __future__ import annotations

"""
Minimal vLLM adapter for text generation.

Best-effort: if vLLM is not installed, raise a clear error. The server can
optionally select this backend by setting backend='vllm'.
"""

from typing import Optional


def generate_text_vllm(prompt: str, max_new_tokens: int = 64, model_id: Optional[str] = None, tensor_parallel_size: int = 1) -> str:

    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"vLLM not available: {e}")

    import os

    model = model_id or os.getenv("OMNICODER_VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    trust = os.getenv("OMNICODER_VLLM_TRUST_REMOTE_CODE", "1") == "1"

    # Reasonable defaults: temperature sampling, short max tokens
    params = SamplingParams(
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        max_tokens=int(max_new_tokens),
    )
    # Construct LLM lazily per call; production code should cache per-process
    llm = LLM(model=model, tensor_parallel_size=int(tensor_parallel_size), trust_remote_code=trust)
    outs = llm.generate([prompt], params)
    if not outs:
        return ""
    try:
        # vLLM returns a RequestOutput list with .outputs list containing .text
        return outs[0].outputs[0].text
    except Exception:
        return ""


