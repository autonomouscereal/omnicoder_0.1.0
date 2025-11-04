from __future__ import annotations

import os
import torch

from omnicoder.sfb.factorize import factorize_prompt
from omnicoder.sfb.inference.bp import BeliefPropagation
from omnicoder.sfb.fusion.cross_bias import CrossBiasFusion


def _dummy_encode(s: str) -> list[int]:
    # Simple toy tokenizer mapping: each char -> ord mod 512
    return [ord(c) % 512 for c in s][:16]


def test_bp_messages_and_cross_bias():
    os.environ['SFB_FACTORIZER'] = 'amr,srl'
    os.environ['SFB_BIAS_ALPHA'] = '1.0'
    prompt = "There is a red cup on the table."
    res = factorize_prompt(prompt)
    bp = BeliefPropagation()
    msgs = bp.run(res.factors)
    # At least one message dict should be produced
    assert isinstance(msgs, list)
    # Apply cross-bias to a fake logits vector
    logits = torch.zeros((1, 1, 1024), dtype=torch.float32)
    fuser = CrossBiasFusion()
    fuser.encode_fn = _dummy_encode
    out = fuser.apply_messages(logits.clone(), msgs)
    assert out.shape == logits.shape
    # Expect some bias applied
    assert torch.any(out != logits)

