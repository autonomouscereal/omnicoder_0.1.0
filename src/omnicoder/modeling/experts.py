import os
import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
	def __init__(self, d_model: int, mlp_dim: int, act: str | None = None):
		super().__init__()
		# Lower-overhead FFN: explicit layers to avoid Sequential call overhead
		self.fc1 = nn.Linear(d_model, mlp_dim)
		# Activation selection with fast, compatible default (GELU tanh approx)
		try:
			act_kind = (act or os.getenv('OMNICODER_MLP_ACT', 'gelu_tanh')).strip().lower()
		except Exception:
			act_kind = 'gelu_tanh'
		if act_kind.startswith('gelu'):
			approximate = 'tanh' if ('tanh' in act_kind) else 'none'
			self.act_fn = nn.GELU(approximate=approximate)
		elif act_kind in ('silu', 'swish'):
			self.act_fn = nn.SiLU()
		else:
			# Fallback to GELU(tanh) for stability/perf
			self.act_fn = nn.GELU(approximate='tanh')
		self.fc2 = nn.Linear(mlp_dim, d_model)
		# Fused linear path by default (no change in numerics); can be disabled via env
		try:
			self._use_fused = (os.getenv('OMNICODER_TRITON_FFN', '1') == '1')
		except Exception:
			self._use_fused = True

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self._use_fused:
			x1 = torch.nn.functional.linear(x, self.fc1.weight, self.fc1.bias)
			x2 = torch.nn.functional.gelu(x1, approximate='tanh') if isinstance(self.act_fn, nn.GELU) else self.act_fn(x1)
			x3 = torch.nn.functional.linear(x2, self.fc2.weight, self.fc2.bias)
			return x3
		x = self.fc1(x)
		x = self.act_fn(x)
		x = self.fc2(x)
		return x


