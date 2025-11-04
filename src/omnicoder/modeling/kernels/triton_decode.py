import os
import torch

try:
	import triton
	import triton.language as tl
	_TRITON_AVAILABLE = True
except Exception:
	_TRITON_AVAILABLE = False


def is_available() -> bool:
	return _TRITON_AVAILABLE and torch.cuda.is_available() and (os.getenv('OMNICODER_TRITON_SEQLEN1','0') == '1')


# Kernel computes: y = softmax(q @ k^T) @ v for seqlen=1
# Shapes:
# - q: (BH, 1, DL)
# - k: (BH, T, DL)
# - v: (BH, T, DL)
# Returns y: (BH, 1, DL)
if _TRITON_AVAILABLE:
	@triton.jit
	def _row_softmax_matmul(
		q_ptr, k_ptr, v_ptr, y_ptr,
		T, DL,
		stride_qb, stride_qt, stride_qd,
		stride_kb, stride_kt, stride_kd,
		stride_vb, stride_vt, stride_vd,
		stride_yb, stride_yt, stride_yd,
		BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
	):
		bid = tl.program_id(axis=0)
		# Each program handles one BH row (single query)
		q_off = bid * stride_qb + 0 * stride_qt
		# Compute logits over keys in tiles of BLOCK_T
		m = tl.full([1], float('-inf'), tl.float32)
		s = tl.zeros([1], tl.float32)
		# First pass: compute max logits for stability
		for t in range(0, T, BLOCK_T):
			cols = t + tl.arange(0, BLOCK_T)
			mask = cols < T
			# Load K tile: (BLOCK_T, DL)
			k_tile = tl.load(k_ptr + bid * stride_kb + cols[:, None] * stride_kt + tl.arange(0, BLOCK_D)[None, :] * stride_kd, mask=mask[:, None], other=0.0)
			q_vec = tl.load(q_ptr + q_off + tl.arange(0, BLOCK_D) * stride_qd)
			# Dot(q, K_i)
			logits = tl.sum(k_tile * q_vec[None, :], axis=1)
			m = tl.maximum(m, tl.max(logits, axis=0))
		# Second pass: compute exp-sum and weighted V
			expsum = tl.zeros([1], tl.float32)
			y_acc = tl.zeros([BLOCK_D], tl.float32)
			for t in range(0, T, BLOCK_T):
				cols = t + tl.arange(0, BLOCK_T)
				mask = cols < T
				k_tile = tl.load(k_ptr + bid * stride_kb + cols[:, None] * stride_kt + tl.arange(0, BLOCK_D)[None, :] * stride_kd, mask=mask[:, None], other=0.0)
				v_tile = tl.load(v_ptr + bid * stride_vb + cols[:, None] * stride_vt + tl.arange(0, BLOCK_D)[None, :] * stride_vd, mask=mask[:, None], other=0.0)
				q_vec = tl.load(q_ptr + q_off + tl.arange(0, BLOCK_D) * stride_qd)
				logits = tl.sum(k_tile * q_vec[None, :], axis=1)
				exp_logits = tl.exp(logits - m)
				expsum += tl.sum(exp_logits, axis=0)
				# Accumulate weighted V
				weights = exp_logits[:, None]
				y_acc += tl.sum(weights * v_tile, axis=0)
			# Normalize
			y = y_acc / (expsum + 1e-9)
			# Store result
			tl.store(y_ptr + bid * stride_yb + 0 * stride_yt + tl.arange(0, BLOCK_D) * stride_yd, y)


def fused_decode_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	"""Fused softmax(bmm) for seqlen=1 using Triton when available.
	q: (BH, 1, DL), k: (BH, T, DL), v: (BH, T, DL) on CUDA, float32/float16
	"""
	if not is_available():
		raise RuntimeError("Triton fused decode not available")
	assert q.is_cuda and k.is_cuda and v.is_cuda
	BH, one, DL = q.shape
	assert one == 1
	T = k.size(1)
	# Ensure contiguous last-dim for striding simplicity
	q = q.contiguous()
	k = k.contiguous()
	v = v.contiguous()
	y = torch.empty((BH, 1, DL), device=q.device, dtype=q.dtype)
	BLOCK_T = 128
	BLOCK_D = min(128, ((DL + 31) // 32) * 32)
	_row_softmax_matmul[
		BH,
	](
		q, k, v, y,
		T, DL,
		q.stride(0), q.stride(1), q.stride(2),
		k.stride(0), k.stride(1), k.stride(2),
		v.stride(0), v.stride(1), v.stride(2),
		y.stride(0), y.stride(1), y.stride(2),
		BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
	)
	return y
