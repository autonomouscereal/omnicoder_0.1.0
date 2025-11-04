import torch
import torch.nn as nn
from typing import Optional
import copy


def upcycle_ffn_to_experts(module: nn.Module, target_experts: int, noise_std: float = 1e-3) -> int:
	"""
	Co-Upcycled MoE: upcycle a dense FFN into multiple experts by cloning the
	base expert weights with small perturbations. This provides a better
	initialization than random for large expert counts.

	Returns number of MoE layers modified.
	"""
	modified = 0
	for name, child in module.named_modules():
		# Identify MoELayer by duck-typing (has 'experts' list of ExpertFFN)
		if hasattr(child, 'experts') and isinstance(getattr(child, 'experts'), nn.ModuleList):
			experts: nn.ModuleList = getattr(child, 'experts')  # type: ignore
			if len(experts) >= target_experts:
				continue
			base = experts[0]
			new_list = nn.ModuleList()
			for i in range(target_experts):
				# Prefer structure-preserving clone to avoid constructor signature coupling
				try:
					clone = copy.deepcopy(base)
				except Exception:
					# Fallback: attempt to reconstruct via dimensions for common ExpertFFN variants
					try:
						# Sequential-based layout: ff[0] is in->mlp, ff[2] is mlp->out
						if hasattr(base, 'ff') and isinstance(getattr(base, 'ff'), nn.Sequential):  # type: ignore[attr-defined]
							d_model = int(base.ff[0].in_features)  # type: ignore[index]
							mlp_dim = int(base.ff[0].out_features)  # type: ignore[index]
							clone = base.__class__(d_model, mlp_dim)  # type: ignore[call-arg]
						elif hasattr(base, 'fc1') and hasattr(base, 'fc2'):
							# Explicit layers layout
							d_model = int(base.fc1.in_features)
							mlp_dim = int(base.fc1.out_features)
							try:
								clone = base.__class__(d_model, mlp_dim, getattr(base, 'act_kind', None))  # type: ignore[call-arg]
							except Exception:
								clone = base.__class__(d_model, mlp_dim)  # type: ignore[call-arg]
						else:
							# Last resort: shallow copy module and re-init parameters
							clone = copy.copy(base)
					except Exception:
						# As a final fallback, use deepcopy again (may raise)
						clone = copy.deepcopy(base)
				try:
					clone.load_state_dict(base.state_dict())  # type: ignore
				except Exception:
					pass
				with torch.no_grad():
					for p in clone.parameters():
						# Avoid autograd version bumps via in-place on leaf params: write to .data
						p.data = (p.data + torch.randn_like(p) * float(noise_std)).clone()
				new_list.append(clone)
			setattr(child, 'experts', new_list)
			# Update metadata if present
			if hasattr(child, 'n_experts'):
				setattr(child, 'n_experts', int(target_experts))
			modified += 1
	return modified


