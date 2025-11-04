#include <torch/extension.h>

// Naive CUDA/ATen implementation: per-(K,E) loops with ATen ops on CUDA tensors
std::tuple<at::Tensor, c10::optional<at::Tensor>> fused_dispatch(
    const at::Tensor& x_flat,
    const at::Tensor& idx_flat_in,
    const at::Tensor& scores_in,
    const torch::List<c10::IValue>& /*experts_unused*/,
    int64_t /*capacity_unused*/,
    const at::Tensor& output_buf,
    const c10::Dict<std::string, at::Tensor>& banks,
    const c10::optional<at::Tensor>& /*hotlog*/,
    const c10::optional<at::Tensor>& /*work_x*/,
    const c10::optional<at::Tensor>& /*work_w*/) {
  TORCH_CHECK(x_flat.dim() == 2, "x_flat must be (N,H)");
  TORCH_CHECK(idx_flat_in.dim() == 2, "idx_flat must be (N,K)");
  TORCH_CHECK(scores_in.dim() == 2, "scores_flat must be (N,K)");

  auto device = x_flat.device();
  auto dtype = x_flat.dtype();
  auto N = x_flat.size(0);
  auto H = x_flat.size(1);
  auto K = idx_flat_in.size(1);

  auto idx_flat = idx_flat_in.to(torch::kLong);
  auto scores = scores_in.to(dtype);

  TORCH_CHECK(banks.contains("W1") && banks.contains("B1") && banks.contains("W2") && banks.contains("B2"),
              "banks must contain W1,B1,W2,B2");
  at::Tensor W1 = banks.at("W1"); // (E,H,M)
  at::Tensor B1 = banks.at("B1"); // (E,M)
  at::Tensor W2 = banks.at("W2"); // (E,M,H)
  at::Tensor B2 = banks.at("B2"); // (E,H)
  TORCH_CHECK(W1.dim() == 3 && W2.dim() == 3 && B1.dim() == 2 && B2.dim() == 2, "invalid bank dims");
  auto E = W1.size(0);
  TORCH_CHECK(W1.size(0) == W2.size(0) && W1.size(0) == B1.size(0) && W2.size(0) == B2.size(0), "bank E mismatch");
  TORCH_CHECK(W1.size(1) == H && W2.size(2) == H, "bank H mismatch");
  auto M = W1.size(2);
  TORCH_CHECK(B1.size(1) == M && W2.size(1) == M, "bank M mismatch");

  at::Tensor out;
  if (output_buf.defined() && output_buf.dim() == 2 && output_buf.size(0) == N && output_buf.size(1) == H && output_buf.device() == device && output_buf.dtype() == dtype) {
    out = output_buf.clone();
    out.zero_();
  } else {
    out = at::zeros({N, H}, x_flat.options());
  }

  for (int64_t k = 0; k < K; ++k) {
    at::Tensor eids = idx_flat.select(1, k);                 // (N)
    at::Tensor gates = scores.select(1, k).unsqueeze(1);     // (N,1)
    for (int64_t e = 0; e < E; ++e) {
      at::Tensor mask = at::eq(eids, e);
      at::Tensor idx = at::nonzero(mask).reshape({-1}).to(torch::kLong); // (Me,)
      at::Tensor x_e = at::index_select(x_flat, 0, idx);     // (Me,H)
      at::Tensor g_e = at::index_select(gates, 0, idx);      // (Me,1)

      at::Tensor W1e = W1.select(0, e); // (H,M)
      at::Tensor B1e = B1.select(0, e); // (M)
      at::Tensor W2e = W2.select(0, e); // (M,H)
      at::Tensor B2e = B2.select(0, e); // (H)

      at::Tensor y1 = at::addmm(B1e, x_e, W1e);
      at::Tensor y1g = at::gelu(y1);
      at::Tensor y2 = at::addmm(B2e, y1g, W2e);             // (Me,H)
      y2 = y2 * g_e;                                        // (Me,H)
      // CG-safe aggregation without accumulate: sort idx, cumsum-diff per token, index_copy
      const auto Me = idx.size(0);
      if (Me > 0) {
        auto sort_res = at::sort(idx);
        at::Tensor idx_sorted = std::get<0>(sort_res);
        at::Tensor order = std::get<1>(sort_res);
        at::Tensor y_sorted = at::index_select(y2, 0, order);
        // positions 0..Me-1
        at::Tensor pos = at::arange(Me, idx.options());
        at::Tensor prev_pos = at::clamp(pos - 1, 0, Me - 1);
        at::Tensor prev_idx = at::index_select(idx_sorted, 0, prev_pos);
        at::Tensor same_prev = at::eq(idx_sorted, prev_idx);
        at::Tensor gt0 = pos > 0;
        at::Tensor is_start = at::logical_not(same_prev.logical_and(gt0));
        // group ends: next is start or last row
        at::Tensor lenm1 = at::full({}, Me - 1, pos.options());
        at::Tensor next_pos = at::minimum(pos + 1, lenm1);
        at::Tensor next_is_start = at::index_select(is_start, 0, next_pos);
        at::Tensor is_last = at::eq(pos, lenm1);
        at::Tensor is_end = at::logical_or(next_is_start, is_last);
        // cumsum over rows, then difference
        at::Tensor y_cum = at::cumsum(y_sorted, 0);
        at::Tensor end_idx_2d = at::nonzero(is_end);
        at::Tensor end_idx = end_idx_2d.reshape({end_idx_2d.size(0)});
        at::Tensor start_idx_2d = at::nonzero(is_start);
        at::Tensor start_idx = start_idx_2d.reshape({start_idx_2d.size(0)});
        at::Tensor start_m1 = at::clamp(start_idx - 1, 0, Me - 1);
        at::Tensor y_end = at::index_select(y_cum, 0, end_idx);
        at::Tensor y_startm1 = at::index_select(y_cum, 0, start_m1);
        // zero-out subtraction for first groups
        at::Tensor is_first_group = at::eq(start_idx, 0);
        at::Tensor mask_first = is_first_group.to(y_startm1.dtype()).unsqueeze(1);
        y_startm1 = y_startm1 * (1 - mask_first);
        at::Tensor seg_sum = y_end - y_startm1; // (Ngroups, H)
        at::Tensor tok_unique = at::index_select(idx_sorted, 0, end_idx);
        out.index_copy_(0, tok_unique, seg_sum);
      }
    }
  }

  return std::make_tuple(out, c10::optional<at::Tensor>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_dispatch", &fused_dispatch, "MoE fused dispatch (CUDA/ATen)");
}


