from __future__ import annotations

import torch
import torch.nn as nn

from .moe_scatter import fused_dispatch as _fused_dispatch

# -------------------------------------------------------------------------------------
# Compiled MoE dispatch wrapper
#
# Why this exists:
# - Calling a free function from the Python module per decode step adds Python overhead.
# - Binding bank tensors as buffers into a tiny nn.Module allows torch.compile to create a
#   stable compiled graph around the call site while keeping the core aten ops inside.
# - We compile once (mode='reduce-overhead'), and reuse the module in the hot loop.
#
# Safety:
# - Banks are optional: if provided at construction time, they are stored as non-persistent
#   buffers; otherwise, call-time banks are passed through to the underlying fused dispatcher.
# - All math remains inside _fused_dispatch; this wrapper does not introduce new side effects.
# -------------------------------------------------------------------------------------


class MoEDispatchModule(nn.Module):

	def __init__(
		self,
		banks: dict[str, torch.Tensor] | None = None,
	):
		super().__init__()
		# Bind banks as buffers if provided so torch.compile can see stable module state
		if banks is not None:
			for k, v in banks.items():
				if isinstance(v, torch.Tensor):
					self.register_buffer(f"bank_{k}", v, persistent=False)
				else:
					setattr(self, f"bank_{k}", None)
		else:
			self.register_buffer("bank_W1", None, persistent=False)
			self.register_buffer("bank_B1", None, persistent=False)
			self.register_buffer("bank_W2", None, persistent=False)
			self.register_buffer("bank_B2", None, persistent=False)

	def forward(
		self,
		x_flat: torch.Tensor,
		idx_flat: torch.Tensor,
		scores_flat: torch.Tensor,
		experts: list | None,
		capacity: int,
		output_buf: torch.Tensor | None = None,
		banks: dict[str, torch.Tensor] | None = None,
		hotlog: torch.Tensor | None = None,
		work_x: torch.Tensor | None = None,
		work_w: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, list[torch.Tensor]]:
		# Prefer bound buffers when available; else use provided dict
		use_banks = banks
		if use_banks is None:
			W1 = getattr(self, "bank_W1", None)
			B1 = getattr(self, "bank_B1", None)
			W2 = getattr(self, "bank_W2", None)
			B2 = getattr(self, "bank_B2", None)
			if (W1 is not None) and (B1 is not None) and (W2 is not None) and (B2 is not None):
				use_banks = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
		return _fused_dispatch(
			x_flat,
			idx_flat,
			scores_flat,
			experts,
			capacity,
			output_buf=output_buf,
			banks=use_banks,
			hotlog=hotlog,
			work_x=work_x,
			work_w=work_w,
		)


def compile_moe_dispatch(banks: dict[str, torch.Tensor] | None = None) -> nn.Module:
	"""Return a compiled MoE dispatch module with optional bound banks.

	The compiled module preserves aten-only ops inside the called fused dispatcher
	and reduces Python overhead in the hot path.
	"""
	# If banks provided, freeze signature to reduce retraces and allow fullgraph compile.
	if banks is not None:
		class MoEDispatchModuleFixed(nn.Module):
			def __init__(self, _banks: dict[str, torch.Tensor]):
				super().__init__()
				# Bind banks as non-persistent buffers
				for k, v in _banks.items():
					self.register_buffer(f"bank_{k}", v, persistent=False)
			def forward(
				self,
				x_flat: torch.Tensor,
				idx_flat: torch.Tensor,
				scores_flat: torch.Tensor,
				capacity: int,
				hotlog: torch.Tensor | None = None,
			) -> tuple[torch.Tensor, list[torch.Tensor]]:
				use_banks = {
					"W1": getattr(self, "bank_W1"),
					"B1": getattr(self, "bank_B1"),
					"W2": getattr(self, "bank_W2"),
					"B2": getattr(self, "bank_B2"),
				}
				# Experts are ignored when banks are bound; pass None to the fused path
				return _fused_dispatch(
					x_flat,
					idx_flat,
					scores_flat,
					experts=None,
					capacity=capacity,
					output_buf=None,
					banks=use_banks,
					hotlog=hotlog,
					work_x=None,
					work_w=None,
				)
		mod: nn.Module = MoEDispatchModuleFixed(banks)
		try:
			compile = getattr(torch, "compile", None)
			if callable(compile):
				# Fullgraph=True with fixed signature reduces retraces
				mod = torch.compile(mod, mode="reduce-overhead", fullgraph=True)  # type: ignore[arg-type]
		except Exception:
			pass
		return mod
	# Fallback: no banks yet, use flexible wrapper
	mod = MoEDispatchModule(banks=banks)
	try:
		compile = getattr(torch, "compile", None)
		if callable(compile):
			mod = torch.compile(mod, mode="reduce-overhead", fullgraph=False)  # type: ignore[arg-type]
	except Exception:
		pass
	return mod


