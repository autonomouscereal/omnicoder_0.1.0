import torch
import torch.nn as nn


class HyenaMixer1D(nn.Module):
    """
    Minimal Hyena-style 1D mixer for long-range sequence modeling.

    Implements a causal, depthwise long convolution with data-dependent gating:
      y = W_o( conv_long( W_in(x) )_a * sigmoid( conv_long( W_in(x) )_b ) )

    Design constraints for compile/export/CG safety:
      - Uses call_module(nn.Conv1d) only; no torch.* functional wrappers in forward
      - Causal via explicit left-pad with aten.constant_pad_nd
      - No Python loops or tensor->Python conversions in hot path

    Input:  x (B, T, C)
    Output: y (B, T, C)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 2,
        kernel_size: int = 256,
        depthwise: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.depthwise = bool(depthwise)

        channels_expanded = self.hidden_dim * self.expansion
        # Pointwise in-projection (C -> C*expansion)
        self.in_proj = nn.Conv1d(self.hidden_dim, channels_expanded, kernel_size=1, bias=True)
        # Depthwise long convolutions for each split half (causal via explicit left-padding)
        c_half = channels_expanded // 2
        groups_half = c_half if self.depthwise else 1
        self.long_conv_a = nn.Conv1d(
            c_half,
            c_half,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups_half,
            bias=True,
        )
        self.long_conv_b = nn.Conv1d(
            c_half,
            c_half,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups_half,
            bias=True,
        )
        # Pointwise out-projection (C*expansion -> C)
        self.out_proj = nn.Conv1d(channels_expanded, self.hidden_dim, kernel_size=1, bias=True)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:  # (batch_size, sequence_length, hidden_dim)
        # Shapes
        batch_size = torch.ops.aten.sym_size.int(input_sequence, 0)
        sequence_length = torch.ops.aten.sym_size.int(input_sequence, 1)
        # Move time to last to channels-first layout expected by Conv1d: (B, C, T)
        input_channels_first = torch.ops.aten.permute.default(
            torch.ops.aten.reshape.default(input_sequence, (batch_size, sequence_length, self.hidden_dim)),
            [0, 2, 1]
        )
        # Pointwise expansion to 2x channels for gated stream construction: (B, C*expansion, T)
        expanded_features = self.in_proj(input_channels_first)
        expanded_channels = torch.ops.aten.sym_size.int(expanded_features, 1)
        half_channels = expanded_channels // 2
        expanded_first_half = torch.ops.aten.slice.Tensor(expanded_features, 1, 0, half_channels, 1)
        expanded_second_half = torch.ops.aten.slice.Tensor(expanded_features, 1, half_channels, expanded_channels, 1)
        # Causal left padding by kernel_size-1 so conv produces length=T
        left_pad = self.kernel_size - 1
        padded_first_half = torch.ops.aten.constant_pad_nd.default(expanded_first_half, (left_pad, 0), 0.0)
        padded_second_half = torch.ops.aten.constant_pad_nd.default(expanded_second_half, (left_pad, 0), 0.0)
        # Long depthwise convolutions along time
        convolved_first_half = self.long_conv_a(padded_first_half)
        convolved_second_half = self.long_conv_b(padded_second_half)
        # Data-dependent gating between halves, then concatenate back to C*expansion
        gate_from_second = torch.ops.aten.sigmoid.default(convolved_second_half)
        gated_first = torch.ops.aten.mul.Tensor(convolved_first_half, gate_from_second)
        gate_from_first = torch.ops.aten.sigmoid.default(convolved_first_half)
        gated_second = torch.ops.aten.mul.Tensor(convolved_second_half, gate_from_first)
        from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
        concatenated_gated = _safe_cat(gated_first, gated_second, 1)  # (B, C*expansion, T)
        # Project back to model hidden dimension and restore (B, T, C)
        projected_channels_first = self.out_proj(concatenated_gated)
        output_sequence = torch.ops.aten.permute.default(projected_channels_first, [0, 2, 1])
        return output_sequence


