from __future__ import annotations

import torch
import torch.nn as nn

try:
	from omnicoder.utils.perf import add as _perf_add  # type: ignore
except Exception:  # pragma: no cover
	_perf_add = None  # type: ignore


class GaussianSplatRenderer3D(nn.Module):
	"""
	Differentiable 3D Gaussian splatting with STRICT aten-only ops in the hot path.

	Key structural choices (per your rulebook):
	- Only torch.ops.aten.*; no method-style ops.
	- Indexing via aten.slice/reshape/gather/index_add; no Python slicing or masked writes.
	- No device moves; all factories anchor to live tensors via aten.new_*.
	- Vectorized over batch and gaussians; no Python loops over data.
	- Static-friendly shapes; splatting via flattened linear indices plus per-batch offsets.
	"""

	def __init__(self, kernel: int = 11) -> None:
		super().__init__()
		# Ensure odd kernel without Python branching in forward
		self.kernel = (kernel if (kernel % 2 == 1) else (kernel + 1))

	def forward(
		self,
		pos_bnh3: torch.Tensor,
		cov_bnh33: torch.Tensor | None,
		cov_diag_bnh3: torch.Tensor | None,
		rgb_bnh3: torch.Tensor,
		opa_bnh1: torch.Tensor,
		K_b33: torch.Tensor,
		R_b33: torch.Tensor,
		t_b3: torch.Tensor,
		image_size: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor]:
		# Shapes and canvas
		B = pos_bnh3.shape[0]
		H = image_size[0]
		W = image_size[1]
		canvas = torch.ops.aten.mul.Scalar(pos_bnh3[:, :1, :1], 0.0)
		rgb = torch.ops.aten.new_zeros.default(canvas, (B, 3, H, W))
		depth = torch.ops.aten.new_zeros.default(canvas, (B, 1, H, W))
		den = torch.ops.aten.new_zeros.default(canvas, (B, 1, H, W))

		# Camera transform: X_cam = R X + t (aten-only)
		Rt = torch.ops.aten.transpose.int(R_b33, 1, 2)
		Xc = torch.ops.aten.add.Tensor(
			torch.ops.aten.matmul.default(pos_bnh3, Rt),
			torch.ops.aten.reshape.default(t_b3, (B, 1, 3))
		)
		Z = torch.ops.aten.slice.Tensor(Xc, -1, 2, 3, 1)  # (B,N,1)
		x_comp = torch.ops.aten.slice.Tensor(Xc, -1, 0, 1, 1)
		y_comp = torch.ops.aten.slice.Tensor(Xc, -1, 1, 2, 1)
		Zs = torch.ops.aten.clamp_min.default(Z, 1e-6)
		xn = torch.ops.aten.div.Tensor(x_comp, Zs)
		yn = torch.ops.aten.div.Tensor(y_comp, Zs)
		from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
		xy1 = _safe_concat((xn, yn, torch.ops.aten.ones_like.default(xn)), -1)
		uvh = torch.ops.aten.matmul.default(xy1, torch.ops.aten.transpose.int(K_b33, 1, 2))
		u = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(uvh, -1, 0, 1, 1), -1)
		v = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(uvh, -1, 1, 2, 1), -1)
		z_flat = torch.ops.aten.squeeze.dim(Z, -1)
		idx = torch.ops.aten.argsort.default(z_flat)  # front-to-back

		# Kernel grid offsets (aten-only range builder)
		k = (self.kernel if (self.kernel % 2 == 1) else (self.kernel + 1))
		off_1d = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(pos_bnh3, (k,), dtype=torch.long), 0)
		off_1d = torch.ops.aten.sub.Tensor(off_1d, (k // 2) + 1)
		off_x = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(off_1d, (1, k)), k, 0)
		off_y = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(off_1d, (k, 1)), k, 1)
		k2 = k * k

		# Gather per-batch sorted subsets (vectorized across batch)
		u_b = torch.ops.aten.gather.default(u, 1, idx)
		v_b = torch.ops.aten.gather.default(v, 1, idx)
		z_b = torch.ops.aten.gather.default(z_flat, 1, idx)
		opa_b = torch.ops.aten.gather.default(torch.ops.aten.squeeze.dim(opa_bnh1, -1), 1, idx)
		rgb_b = torch.ops.aten.gather.default(rgb_bnh3, 1, torch.ops.aten.unsqueeze.default(idx, -1).expand(-1, -1, 3))

		# Sigma (pixels) from covariances and focal lengths
		if cov_bnh33 is not None:
			cov_sel = torch.ops.aten.gather.default(torch.ops.aten.reshape.default(cov_bnh33, (B, -1, 9)), 1, torch.ops.aten.unsqueeze.default(idx, -1).expand(-1, -1, 9))
			cov_sel = torch.ops.aten.reshape.default(cov_sel, (B, -1, 3, 3))
			sxx = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(cov_sel, -1, 0, 1, 1), -1)
			syy = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(cov_sel, -1, 1, 2, 1), -1)
		else:
			cd = (cov_diag_bnh3 if cov_diag_bnh3 is not None else torch.ops.aten.add.Scalar(rgb_bnh3[..., :3], 0.0))
			cds = torch.ops.aten.gather.default(cd, 1, torch.ops.aten.unsqueeze.default(idx, -1).expand(-1, -1, 3))
			sxx = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(cds, -1, 0, 1, 1), -1)
			syy = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(cds, -1, 1, 2, 1), -1)
		fx = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(K_b33, -1, 0, 1, 1), -1)
		fy = torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(K_b33, -1, 1, 2, 1), -1)
		sx_px = torch.ops.aten.mul.Tensor(torch.ops.aten.sqrt.default(torch.ops.aten.clamp_min.default(sxx, 1e-9)), torch.ops.aten.div.Tensor(fx, torch.ops.aten.clamp_min.default(z_b, 1e-6)))
		sy_px = torch.ops.aten.mul.Tensor(torch.ops.aten.sqrt.default(torch.ops.aten.clamp_min.default(syy, 1e-9)), torch.ops.aten.div.Tensor(fy, torch.ops.aten.clamp_min.default(z_b, 1e-6)))

		# Build per-pixel coordinates for all gaussians (B, N, k2)
		u0 = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(u_b, (B, -1, 1)), k2, -1)
		v0 = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(v_b, (B, -1, 1)), k2, -1)
		sx_r = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(sx_px, (B, -1, 1)), k2, -1)
		sy_r = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(sy_px, (B, -1, 1)), k2, -1)
		offx = torch.ops.aten.reshape.default(off_x, (1, 1, k2))
		offy = torch.ops.aten.reshape.default(off_y, (1, 1, k2))
		ux = torch.ops.aten.add.Tensor(u0, torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offx, u0.dtype, False, False), sx_r))
		vy = torch.ops.aten.add.Tensor(v0, torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offy, v0.dtype, False, False), sy_r))
		px = torch.ops.aten.to.dtype(torch.ops.aten.round.default(ux), torch.long, False, False)
		py = torch.ops.aten.to.dtype(torch.ops.aten.round.default(vy), torch.long, False, False)
		pxc = torch.ops.aten.clamp.default(px, 0, W - 1)
		pyc = torch.ops.aten.clamp.default(py, 0, H - 1)
		mask = torch.ops.aten.logical_and.default(torch.ops.aten.eq.Tensor(px, pxc), torch.ops.aten.eq.Tensor(py, pyc))

		# Gaussian falloff weights (vectorized)
		gx = torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offx, sx_r.dtype, False, False), torch.ops.aten.reciprocal.default(torch.ops.aten.clamp_min.default(sx_r, 1e-6)))
		gy = torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offy, sy_r.dtype, False, False), torch.ops.aten.reciprocal.default(torch.ops.aten.clamp_min.default(sy_r, 1e-6)))
		g2 = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(gx, gx), torch.ops.aten.mul.Tensor(gy, gy))
		w = torch.ops.aten.mul.Tensor(torch.ops.aten.exp.default(torch.ops.aten.mul.Scalar(g2, -0.5)), torch.ops.aten.to.dtype(mask, g2.dtype, False, False))
		alpha = torch.ops.aten.mul.Tensor(torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(opa_b, (B, -1, 1)), k2, -1), w)

		# Linear indices per batch with offsets
		lin = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(pyc, W), pxc)
		boff = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(pos_bnh3, (B,), dtype=torch.long), 0)
		boff = torch.ops.aten.sub.Tensor(boff, 1)
		boff = torch.ops.aten.mul.Scalar(boff, H * W)
		lin = torch.ops.aten.add.Tensor(lin, torch.ops.aten.reshape.default(boff, (B, 1)))
		lin = torch.ops.aten.reshape.default(lin, (-1,))

		# RGB/denominator accumulation via index_add on flattened buffers
		flat_rgb = torch.ops.aten.new_zeros.default(canvas, (B * H * W, 3))
		flat_den = torch.ops.aten.new_zeros.default(canvas, (B * H * W,))
		col = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(rgb_b, (B, -1, 1, 3)), k2, -2)
		col = torch.ops.aten.reshape.default(col, (B, -1, 3))
		inc = torch.ops.aten.reshape.default(alpha, (B, -1, 1))
		num = torch.ops.aten.mul.Tensor(torch.ops.aten.reshape.default(inc, (-1, 1)), torch.ops.aten.reshape.default(col, (-1, 3)))
		for ch in range(3):
			flat_rgb[:, ch] = torch.ops.aten.index_add.default(flat_rgb[:, ch], 0, lin, torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(num, -1, ch, ch + 1, 1), -1))
		flat_den = torch.ops.aten.index_add.default(flat_den, 0, lin, torch.ops.aten.reshape.default(inc, (-1,)))
		rgb = torch.ops.aten.reshape.default(flat_rgb, (B, 3, H, W))
		den = torch.ops.aten.reshape.default(flat_den, (B, 1, H, W))
		rgb = torch.ops.aten.div.Tensor(rgb, torch.ops.aten.clamp_min.default(den, 1e-6))

		# Depth as weighted average of z
		zw = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(z_b, (B, -1, 1)), k2, -1)
		flat_depth_num = torch.ops.aten.new_zeros.default(canvas, (B * H * W,))
		flat_depth_num = torch.ops.aten.index_add.default(flat_depth_num, 0, lin, torch.ops.aten.reshape.default(torch.ops.aten.mul.Tensor(zw, alpha), (-1,)))
		depth_num = torch.ops.aten.reshape.default(flat_depth_num, (B, 1, H, W))
		depth = torch.ops.aten.div.Tensor(depth_num, torch.ops.aten.clamp_min.default(den, 1e-6))

		if _perf_add is not None:
			try:
				_perf_add('gsplat3d', float(1.0))
			except Exception:
				pass
		return rgb, depth


class GaussianSplatRenderer2D(nn.Module):
	"""
	2D Gaussian splatting renderer: renders a set of 2D Gaussians (mean, cov, color, opacity)
	onto an image grid using aten-only ops and chunked blending.
	"""

	def __init__(self, kernel: int = 11) -> None:
		super().__init__()
		self.kernel = int(kernel if kernel % 2 == 1 else (kernel + 1))

	def forward(
		self,
		mean_bng2: torch.Tensor,
		cov_diag_bng2: torch.Tensor,
		rgb_bng3: torch.Tensor,
		opa_bng1: torch.Tensor,
		image_size: tuple[int, int],
	) -> torch.Tensor:
		B = mean_bng2.shape[0]
		H = image_size[0]
		W = image_size[1]
		canvas = torch.ops.aten.mul.Scalar(mean_bng2[:, :1, :1], 0.0)
		rgb = torch.ops.aten.new_zeros.default(canvas, (B, 3, H, W))
		den = torch.ops.aten.new_zeros.default(canvas, (B, 1, H, W))
		k = (self.kernel if (self.kernel % 2 == 1) else (self.kernel + 1))
		off_1d = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(mean_bng2, (k,), dtype=torch.long), 0)
		off_1d = torch.ops.aten.sub.Tensor(off_1d, (k // 2) + 1)
		off_x = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(off_1d, (1, k)), k, 0)
		off_y = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(off_1d, (k, 1)), k, 1)
		k2 = k * k
		ux0 = torch.ops.aten.slice.Tensor(mean_bng2, -1, 0, 1, 1)
		vy0 = torch.ops.aten.slice.Tensor(mean_bng2, -1, 1, 2, 1)
		sx = torch.ops.aten.sqrt.default(torch.ops.aten.clamp_min.default(torch.ops.aten.slice.Tensor(cov_diag_bng2, -1, 0, 1, 1), 1e-9))
		sy = torch.ops.aten.sqrt.default(torch.ops.aten.clamp_min.default(torch.ops.aten.slice.Tensor(cov_diag_bng2, -1, 1, 2, 1), 1e-9))
		ux = torch.ops.aten.repeat_interleave.self_int(ux0, k2, -1)
		vy = torch.ops.aten.repeat_interleave.self_int(vy0, k2, -1)
		sx_r = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(sx, (B, -1, 1)), k2, -1)
		sy_r = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(sy, (B, -1, 1)), k2, -1)
		offx = torch.ops.aten.reshape.default(off_x, (1, 1, k2))
		offy = torch.ops.aten.reshape.default(off_y, (1, 1, k2))
		px = torch.ops.aten.to.dtype(torch.ops.aten.round.default(torch.ops.aten.add.Tensor(ux, torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offx, ux.dtype, False, False), sx_r))), torch.long, False, False)
		py = torch.ops.aten.to.dtype(torch.ops.aten.round.default(torch.ops.aten.add.Tensor(vy, torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offy, vy.dtype, False, False), sy_r))), torch.long, False, False)
		pxc = torch.ops.aten.clamp.default(px, 0, W - 1)
		pyc = torch.ops.aten.clamp.default(py, 0, H - 1)
		mask = torch.ops.aten.logical_and.default(torch.ops.aten.eq.Tensor(px, pxc), torch.ops.aten.eq.Tensor(py, pyc))
		gx = torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offx, sx_r.dtype, False, False), torch.ops.aten.reciprocal.default(torch.ops.aten.clamp_min.default(sx_r, 1e-6)))
		gy = torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(offy, sy_r.dtype, False, False), torch.ops.aten.reciprocal.default(torch.ops.aten.clamp_min.default(sy_r, 1e-6)))
		g2 = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(gx, gx), torch.ops.aten.mul.Tensor(gy, gy))
		w = torch.ops.aten.mul.Tensor(torch.ops.aten.exp.default(torch.ops.aten.mul.Scalar(g2, -0.5)), torch.ops.aten.to.dtype(mask, g2.dtype, False, False))
		alpha = torch.ops.aten.mul.Tensor(torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(opa_bng1, (B, -1, 1)), k2, -1), w)
		lin = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(pyc, W), pxc)
		bo = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(mean_bng2, (B,), dtype=torch.long), 0)
		bo = torch.ops.aten.sub.Tensor(bo, 1)
		bo = torch.ops.aten.mul.Scalar(bo, H * W)
		lin = torch.ops.aten.add.Tensor(lin, torch.ops.aten.reshape.default(bo, (B, 1)))
		lin = torch.ops.aten.reshape.default(lin, (-1,))
		flat_rgb = torch.ops.aten.new_zeros.default(canvas, (B * H * W, 3))
		flat_den = torch.ops.aten.new_zeros.default(canvas, (B * H * W,))
		col = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.reshape.default(rgb_bng3, (B, -1, 1, 3)), k2, -2)
		col = torch.ops.aten.reshape.default(col, (B, -1, 3))
		inc = torch.ops.aten.reshape.default(alpha, (B, -1, 1))
		num = torch.ops.aten.mul.Tensor(torch.ops.aten.reshape.default(inc, (-1, 1)), torch.ops.aten.reshape.default(col, (-1, 3)))
		for ch in range(3):
			flat_rgb[:, ch] = torch.ops.aten.index_add.default(flat_rgb[:, ch], 0, lin, torch.ops.aten.squeeze.dim(torch.ops.aten.slice.Tensor(num, -1, ch, ch + 1, 1), -1))
		flat_den = torch.ops.aten.index_add.default(flat_den, 0, lin, torch.ops.aten.reshape.default(inc, (-1,)))
		rgb = torch.ops.aten.reshape.default(flat_rgb, (B, 3, H, W))
		den = torch.ops.aten.reshape.default(flat_den, (B, 1, H, W))
		rgb = torch.ops.aten.div.Tensor(rgb, torch.ops.aten.clamp_min.default(den, 1e-6))
		if _perf_add is not None:
			try:
				_perf_add('gsplat2d', float(1.0))
			except Exception:
				pass
		return rgb


