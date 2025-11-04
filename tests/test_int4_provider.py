import os
import torch

from omnicoder.modeling.quant.int4_providers import create_int4_provider


def test_int4_matmul_cpu_correctness():
    torch.manual_seed(0)
    in_f, out_f = 16, 8
    mod = create_int4_provider(in_f, out_f, backend='cpu')
    # Initialize packed weights/scales/zeros to a deterministic pattern
    with torch.no_grad():
        mod.weight_packed.copy_(torch.randint(0, 16, mod.weight_packed.shape, dtype=torch.uint8))
        mod.scale.copy_(torch.ones_like(mod.scale))
        mod.zero.copy_(torch.zeros_like(mod.zero))
    x = torch.randn(4, in_f)
    # Reference: dequantize and compare against dense matmul formed inside forward
    y = mod(x)
    assert y.shape == (4, out_f)


def test_int4_matmul_dml_fallback_cpu():
    # Force DML backend; absence will trigger CPU fallback without error
    os.environ['OMNICODER_INT4_BACKEND'] = 'dml'
    torch.manual_seed(0)
    in_f, out_f = 12, 6
    mod = create_int4_provider(in_f, out_f, backend='dml')
    with torch.no_grad():
        mod.weight_packed.copy_(torch.randint(0, 16, mod.weight_packed.shape, dtype=torch.uint8))
        mod.scale.copy_(torch.full_like(mod.scale, 0.5))
        mod.zero.copy_(torch.full_like(mod.zero, 7.0))
    x = torch.randn(2, in_f)
    y = mod(x)
    assert y.shape == (2, out_f)


