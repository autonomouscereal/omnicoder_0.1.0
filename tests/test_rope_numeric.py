import torch

from omnicoder.modeling.attention import _apply_rotary_pos_emb


def _naive_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # x: (B,H,T,D), sin/cos: (T,D)
    b,h,t,d = x.shape
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    sin_t = sin[:t, :d].unsqueeze(0).unsqueeze(0)
    cos_t = cos[:t, :d].unsqueeze(0).unsqueeze(0)
    # Interleave even/odd rotations
    xe = x1 * cos_t[..., ::2] - x2 * sin_t[..., ::2]
    xo = x2 * cos_t[..., ::2] + x1 * sin_t[..., ::2]
    y = torch.stack((xe, xo), dim=-1).reshape(b,h,t,d)
    return y


def test_rope_numeric_match():
    torch.manual_seed(0)
    B,H,T,D = 1, 4, 8, 16
    x = torch.randn(B,H,T,D)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    t_idx = torch.arange(T).float()
    freqs = torch.einsum('t,f->tf', t_idx, inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)
    sin = torch.sin(rope)
    cos = torch.cos(rope)
    y_ref = _naive_rope(x, sin, cos)
    y = _apply_rotary_pos_emb(x, sin, cos)
    assert torch.allclose(y, y_ref, atol=1e-6)

