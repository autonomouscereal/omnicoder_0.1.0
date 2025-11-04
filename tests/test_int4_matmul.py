import torch

from omnicoder.modeling.kernels import omnicoder_dml_op  # ensure loader tries to load native ops
from omnicoder.modeling.quant.int4_providers import Int4MatmulCPU


def test_int4_matmul_shapes():
    m = Int4MatmulCPU(in_features=16, out_features=8)
    # Initialize a trivial packed weight to zeros
    with torch.no_grad():
        m.weight_packed.zero_()
        m.scale.fill_(0.1)
        m.zero.fill_(0.0)
    x = torch.ones(2, 16)
    y = m(x)
    assert y.shape == (2, 8)


