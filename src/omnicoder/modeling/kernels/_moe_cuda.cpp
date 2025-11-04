#include <torch/extension.h>

// Forward declaration of CUDA kernel launcher
torch::Tensor moe_dispatch_forward_cuda(
    const torch::Tensor& x_flat,      // (N, C)
    const torch::Tensor& idx_flat,    // (N, K) int64
    const torch::Tensor& scores_flat, // (N, K) float32
    const torch::Tensor& expert_offsets, // (E+1,) int64 cumulative counts
    const torch::Tensor& token_indices,  // (M,) int64 gathered token indices for all experts
    int64_t hidden_dim,
    int64_t capacity);


torch::Tensor moe_dispatch_forward(
    const torch::Tensor& x_flat,
    const torch::Tensor& idx_flat,
    const torch::Tensor& scores_flat,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& token_indices,
    int64_t hidden_dim,
    int64_t capacity) {
  TORCH_CHECK(x_flat.is_cuda(), "x_flat must be CUDA");
  TORCH_CHECK(idx_flat.is_cuda(), "idx_flat must be CUDA");
  TORCH_CHECK(scores_flat.is_cuda(), "scores_flat must be CUDA");
  return moe_dispatch_forward_cuda(x_flat, idx_flat, scores_flat, expert_offsets, token_indices, hidden_dim, capacity);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_dispatch_forward", &moe_dispatch_forward, "Fused MoE dispatch forward (CUDA)");
}


